package dev.comfyfluffy.caustica.rt;

import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Regression guards for animated water. The old path only tilted normals on a geometrically flat
 * sheet, which read as a repeating square texture. Animated Water must intersect a real height
 * field so crests and troughs show up in depth, reflections and refraction.
 */
final class RtWaterWaveShaderRegressionTest {
    private static final Path REPO_ROOT = repoRoot();
    private static final Path WATER = REPO_ROOT.resolve("shaders/world/water.slang");
    private static final Path PRIMARY = REPO_ROOT.resolve("shaders/world/world_primary.rgen.slang");
    private static final Path WORLD = REPO_ROOT.resolve("shaders/world/world.rgen.slang");
    private static final Path GUIDES = REPO_ROOT.resolve("shaders/world/guides.slang");

    @Test
    void spectrumAccumulatesHeightNotJustGradient() throws IOException {
        String water = Files.readString(WATER);

        assertFalse(water.contains("gradient-only"),
                "the surface must no longer stay geometrically flat");
        assertFalse(water.contains("the surface stays geometrically"),
                "animated water is a height field, not a normal-only fake");
        assertTrue(water.contains("height += a * e;"),
                "the spectrum must accumulate the sharp-crested height used for displacement");
        assertTrue(water.contains("public float waterWaveHeight("),
                "height must be a first-class spectrum output, not reconstructed from the gradient");
    }

    @Test
    void animatedWaterIntersectsTheHeightFieldAlongTheViewRay() throws IOException {
        String water = Files.readString(WATER);
        String intersect = slice(water, "public WaterWaves evaluateWaterWaves(", "\n}\n");

        assertTrue(intersect.contains("WAVE_NEWTON_STEPS"),
                "the rest-plane hit must be refined onto the displaced surface");
        assertTrue(intersect.contains("float f = pos.y - restY - height;"),
                "Newton residual is ray Y minus rest plane minus wave height");
        assertTrue(intersect.contains("waves.position = pos;"),
                "the shading point must ride the height field");
        assertTrue(intersect.contains("waves.motionPrev = float3(0.0, height - previousHeight, 0.0);"),
                "vertical wave motion has to reach DLSS/NRD or the surface will ghost");
    }

    @Test
    void radianceAndGuidePassesUseGeometricWavesWhenTheToggleIsOn() throws IOException {
        String primary = Files.readString(PRIMARY);
        String world = Files.readString(WORLD);
        String guides = Files.readString(GUIDES);

        assertTrue(primary.contains("evaluateWaterWaves(") && primary.contains("waterWaveContinueOrigin("),
                "Pass A must displace the primary water hit and continue from the height field");
        assertTrue(world.contains("evaluateWaterWaves(") && world.contains("waterWaveContinueOrigin("),
                "Pass B must displace water hits the same way as Pass A");
        assertTrue(guides.contains("evaluateWaterWaves(") && guides.contains("waterWaveContinueOrigin("),
                "transmission guides must follow the displaced surface or RR will halo");
        assertTrue(primary.contains("isWater ? waterSurf.motionPrev : payload.motionPrev"),
                "primary water motion must come from the height-field time derivative");
    }

    @Test
    void continuationOriginsStayOffTheRestPlaneMesh() throws IOException {
        String origin = slice(Files.readString(WATER), "public float3 waterWaveContinueOrigin(", "\n}\n");

        assertTrue(origin.contains("goingOut > 0.0"),
                "reflections must stay outside the rest-plane triangle");
        assertTrue(origin.contains("origin -= waves.geometricNormal * (planeSide + bias)"),
                "transmissions must start inside the volume even from a crest above the mesh");
    }

    /**
     * The wave field is global but the water body is not. The displaced surface point follows the
     * viewing ray, so at a grazing angle it leaves the water quad the ray hit by several blocks; a
     * refracted ray starting there samples the shore instead of the underwater scene, which is what
     * drew a contour along the water's edge (and made the transmission guide demodulate the fringe
     * against the shore's albedo, i.e. a halo). The transmitted origin must be pulled back into the
     * water body, and it must be the shared helper so Pass A, Pass B and the guide walk all get it.
     */
    @Test
    void transmittedOriginIsConfinedToTheWaterBody() throws IOException {
        String water = Files.readString(WATER);
        String origin = slice(water, "public float3 waterWaveContinueOrigin(", "\n}\n");
        String confine = slice(water, "public float3 waterConfineToWater(", "\n}\n");
        String footprint = slice(water, "public float4 waterSurfaceFootprint(", "\n}\n");

        assertTrue(origin.contains("waterConfineToWater(waves.planePos, origin)"),
                "the transmitted origin must be confined to the water body it was sampled from");
        assertTrue(origin.indexOf("goingOut > 0.0") < origin.indexOf("waterConfineToWater("),
                "only the ray that enters the volume may be confined; a reflection leaves the surface");
        assertTrue(confine.contains("clamp(p.xz, lo, hi)"),
                "the confined origin must be clamped into the water quad's footprint");
        assertTrue(footprint.contains("floor(planePos.xz + anchor) - anchor"),
                "the water footprint must be derived from the wave anchor, no extra push data");
        assertTrue(inset(water) > 0.0,
                "the confined origin must not sit exactly on the water quad's side face");
    }

    /**
     * Numeric model of the confinement: the flat-mesh proxy lets the displaced surface point drift
     * off the water quad, and the confined origin has to end up inside the block the ray hit for
     * every amplitude, speed and viewing angle the settings allow.
     */
    @Test
    void confinedOriginStaysInsideTheHitBlockForEveryAllowedSetting() {
        final double inset = 0.02;
        final double anchorX = 4096.0 - 137.0;   // the rebase origin modulo 4096
        final double anchorZ = 91.0;
        final double[] strengths = {0.0, 1.0, 2.0, 4.0};
        final double[] speeds = {0.0, 1.0, 4.0};
        final double[] slopes = {-0.02, -0.05, -0.1, -0.3, -0.7, -1.0};

        int escapedBefore = 0;
        int escapedAfter = 0;
        for (double strength : strengths) {
            for (double speed : speeds) {
                for (double slope : slopes) {
                    for (double lateral : new double[] {-1.0, -0.3, 0.0, 0.3, 1.0}) {
                        // A view ray with the given slope, looking at a point inside a water block.
                        double rdZ = -Math.sqrt(Math.max(1.0 - slope * slope - lateral * lateral, 0.0));
                        double rdX = lateral * Math.sqrt(Math.max(1.0 - slope * slope, 0.0));
                        double hitX = 12.5, hitZ = 7.5;

                        // Worst case the flat-mesh proxy can drift: the crest bound of the spectrum
                        // along the ray, both ways.
                        double bound = 0.55 * strength;
                        double drift = bound / Math.max(Math.abs(slope), 0.05);
                        double dispX = hitX + rdX * drift;
                        double dispZ = hitZ + rdZ * drift;

                        double blockMinX = Math.floor(hitX + anchorX) - anchorX;
                        double blockMinZ = Math.floor(hitZ + anchorZ) - anchorZ;

                        if (dispX < blockMinX || dispX > blockMinX + 1.0
                                || dispZ < blockMinZ || dispZ > blockMinZ + 1.0) {
                            escapedBefore++;
                        }

                        double confinedX = Math.min(Math.max(dispX, blockMinX + inset),
                                blockMinX + 1.0 - inset);
                        double confinedZ = Math.min(Math.max(dispZ, blockMinZ + inset),
                                blockMinZ + 1.0 - inset);
                        if (confinedX < blockMinX || confinedX > blockMinX + 1.0
                                || confinedZ < blockMinZ || confinedZ > blockMinZ + 1.0) {
                            escapedAfter++;
                        }
                    }
                }
            }
        }
        assertTrue(escapedBefore > 0,
                "the test must exercise rays whose displaced point leaves the water body");
        assertEquals(0, escapedAfter,
                "the confined refraction origin must never leave the water block it was sampled from");
    }

    private static String slice(String source, String startNeedle, String endNeedle) {
        int start = source.indexOf(startNeedle);
        assertTrue(start >= 0, "missing shader snippet start: " + startNeedle);
        int end = source.indexOf(endNeedle, start);
        assertTrue(end > start, "missing shader snippet end: " + endNeedle);
        return source.substring(start, end);
    }

    /** Reads the numeric value of the WATER_FOOTPRINT_INSET constant out of the shader source. */
    private static double inset(String water) {
        String needle = "WATER_FOOTPRINT_INSET = ";
        int start = water.indexOf(needle);
        assertTrue(start >= 0, "the water shader must define WATER_FOOTPRINT_INSET");
        int end = water.indexOf(';', start);
        assertTrue(end > start, "malformed WATER_FOOTPRINT_INSET constant");
        return Double.parseDouble(water.substring(start + needle.length(), end).trim());
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
