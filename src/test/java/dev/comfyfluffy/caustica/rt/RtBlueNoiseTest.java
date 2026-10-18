package dev.comfyfluffy.caustica.rt;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Guards the blue-noise dither tile and its shader plumbing.
 *
 * <p>The tile is the only new renderer input in the blue-noise change, and its two properties matter
 * for correctness rather than taste: every 8-bit channel must stay an exactly uniform permutation (an
 * uneven channel would bias every BSDF and light sample) and neighbouring texels must stay
 * anti-correlated (the whole point over white noise). The source-level assertions then pin the
 * transport — the tile rides the BDA ring slot, is written once per slot, and the raygen actually
 * consumes it for the NEE and BSDF sample pairs while keeping the old PCG draws as the fallback.
 */
final class RtBlueNoiseTest {
    private static final Path REPO_ROOT = repoRoot();
    private static final Path WORLD_COMMON = REPO_ROOT.resolve("shaders/world/world_common.slang");
    private static final Path WORLD_CORE = REPO_ROOT.resolve("shaders/world/world_core.slang");
    private static final Path MATH = REPO_ROOT.resolve("shaders/world/math.slang");
    private static final Path WORLD_RGEN = REPO_ROOT.resolve("shaders/world/world.rgen.slang");
    private static final Path RT_COMPOSITE =
            REPO_ROOT.resolve("src/main/java/dev/comfyfluffy/caustica/rt/RtComposite.java");

    @Test
    void tileShapeMatchesTheShaderMask() {
        int[] words = RtBlueNoise.INSTANCE.words();
        assertEquals(RtBlueNoise.TILE * RtBlueNoise.TILE, words.length,
                "the tile must stay square and power-of-two per side");
        assertEquals(0, RtBlueNoise.TILE & (RtBlueNoise.TILE - 1),
                "the shader masks pixel coordinates with TILE - 1, which needs a power of two");
        assertEquals(RtBlueNoise.WORDS * 4, RtBlueNoise.BYTES,
                "one packed RGBA8 word per texel");
    }

    /** An uneven channel would bias whichever decision samples it; every level must appear equally. */
    @Test
    void everyChannelIsFlatOverAllByteLevels() {
        int[] words = RtBlueNoise.INSTANCE.words();
        int expected = RtBlueNoise.TILE * RtBlueNoise.TILE / 256;
        for (int channel = 0; channel < 4; channel++) {
            int shift = channel * 8;
            int[] histogram = new int[256];
            for (int word : words) {
                histogram[(word >>> shift) & 0xFF]++;
            }
            for (int level = 0; level < 256; level++) {
                assertEquals(expected, histogram[level],
                        "channel " + channel + " level " + level + " is not equally likely");
            }
        }
    }

    /** Blue noise = high-frequency energy: adjacent texels must disagree, unlike white noise. */
    @Test
    void neighbouringTexelsAreAntiCorrelated() {
        int[] words = RtBlueNoise.INSTANCE.words();
        for (int channel = 0; channel < 4; channel++) {
            double correlation = lagOneCorrelation(words, channel);
            assertTrue(correlation < -0.15,
                    "channel " + channel + " lag-1 correlation is " + correlation
                            + "; a blue-noise mask must be clearly anti-correlated with its neighbours");
        }
    }

    @Test
    void generationIsDeterministic() {
        int[] first = RtBlueNoise.INSTANCE.words();
        int[] second = RtBlueNoise.INSTANCE.words();
        assertSame(first, second, "the tile is generated once and cached for the session");
    }

    @Test
    void tileTravelsToTheShaderThroughThePushRingSlot() throws IOException {
        String common = Files.readString(WORLD_COMMON);
        assertTrue(common.contains("public uint64_t blueNoiseAddr;"),
                "WorldPush needs the blue-noise tile address the shader reads");

        String core = Files.readString(WORLD_CORE);
        assertTrue(core.contains("return worldPush.blueNoiseAddr != 0;"),
                "blueNoiseEnabled must be the single hard fallback check");

        String composite = Files.readString(RT_COMPOSITE);
        assertTrue(composite.contains("BLUE_NOISE_OFFSET + RtBlueNoise.BYTES"),
                "the ring slot must reserve room for the tile");
        assertTrue(composite.contains("writeBlueNoise(slot.buffer);"),
                "every ring slot must receive the tile when it is created");
        assertTrue(composite.contains("pushBuf.deviceAddress + BLUE_NOISE_OFFSET,"),
                "the per-frame push must carry the tile address");
    }

    @Test
    void raygenDrawsBothSamplePairsFromTheTileAndKeepsPcgAsFallback() throws IOException {
        String math = Files.readString(MATH);
        assertTrue(math.contains("public float4 blueNoiseTexel(uint2 pixel, uint salt)"),
                "math.slang must expose the tile fetch");
        assertTrue(math.contains("tileY * uint(BLUE_NOISE_TILE) + tileX"),
                "the fetch must wrap inside the tile");
        assertTrue(math.contains("worldPush.frameIndex * 11u + salt * 7u"),
                "the tile must be temporally rotated per frame and sample");
        assertTrue(math.contains("public float3 cosineDir(float3 n, float u1, float u2)")
                        && math.contains("public float3 sampleSquare(float3 axis, float halfAngle, float u, float v)")
                        && math.contains("public float3 sampleGGXVNDF(float3 n, float3 ve, float alpha, float u1, float u2)"),
                "the samplers need explicit-pair overloads so blue-noise components can drive them");

        String rgen = Files.readString(WORLD_RGEN);
        assertTrue(rgen.contains("float4 blueNoise = blueNoiseTexel("),
                "the indirect pass must fetch one tile texel per shading vertex");
        assertTrue(rgen.contains("sampleSquare(lightDir, lightHalfAngle, blueNoise.z, blueNoise.w)"),
                "the NEE light position must come from the tile's B/A pair");
        assertTrue(rgen.contains("sampleGGXVNDF(n, v, rough, blueNoise.x, blueNoise.y)")
                        && rgen.contains("cosineDir(n, blueNoise.x, blueNoise.y)"),
                "both BSDF continuation lobes must come from the tile's R/G pair");
        // The fallback draws must survive: with no tile published the renderer keeps the PCG stream.
        assertTrue(rgen.contains(": sampleSquare(lightDir, lightHalfAngle, seed)"),
                "the NEE sample must fall back to the PCG stream");
        assertTrue(rgen.contains(": sampleGGXVNDF(n, v, rough, seed)")
                        && rgen.contains(": cosineDir(n, seed)"),
                "the BSDF samples must fall back to the PCG stream");
    }

    /** Pearson correlation between a channel and its right neighbour, wrapping across the tile edge. */
    private static double lagOneCorrelation(int[] words, int channel) {
        List<Double> left = new ArrayList<>(words.length);
        List<Double> right = new ArrayList<>(words.length);
        int shift = channel * 8;
        for (int y = 0; y < RtBlueNoise.TILE; y++) {
            for (int x = 0; x < RtBlueNoise.TILE; x++) {
                int here = (words[y * RtBlueNoise.TILE + x] >>> shift) & 0xFF;
                int next = (words[y * RtBlueNoise.TILE + ((x + 1) & (RtBlueNoise.TILE - 1))] >>> shift) & 0xFF;
                left.add((double) here);
                right.add((double) next);
            }
        }
        double meanLeft = left.stream().mapToDouble(Double::doubleValue).average().orElseThrow();
        double meanRight = right.stream().mapToDouble(Double::doubleValue).average().orElseThrow();
        double covariance = 0.0;
        double varianceLeft = 0.0;
        double varianceRight = 0.0;
        for (int i = 0; i < left.size(); i++) {
            double dl = left.get(i) - meanLeft;
            double dr = right.get(i) - meanRight;
            covariance += dl * dr;
            varianceLeft += dl * dl;
            varianceRight += dr * dr;
        }
        return covariance / Math.sqrt(varianceLeft * varianceRight);
    }

    private static Path repoRoot() {
        Path dir = Path.of("").toAbsolutePath();
        for (Path candidate = dir; candidate != null; candidate = candidate.getParent()) {
            if (Files.isDirectory(candidate.resolve("shaders/world"))
                    && Files.isDirectory(candidate.resolve("src/main/java"))) {
                return candidate;
            }
        }
        throw new IllegalStateException("could not locate the repository root from " + dir);
    }
}
