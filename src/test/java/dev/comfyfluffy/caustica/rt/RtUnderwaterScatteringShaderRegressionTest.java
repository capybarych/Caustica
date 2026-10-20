package dev.comfyfluffy.caustica.rt;

import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Guards the underwater path against black chunk-boundary misses and depth-dependent haze tint. */
final class RtUnderwaterScatteringShaderRegressionTest {
    @Test
    void underwaterHazeUsesRealLightWithStableBlueChromaticity() throws IOException {
        Path root = repoRoot();
        String medium = Files.readString(root.resolve("shaders/world/medium.slang"));
        String world = Files.readString(root.resolve("shaders/world/world.rgen.slang"));
        assertTrue(medium.contains("public float3 waterHazeScattering(float3 rayDir, float segmentDistance)"));
        assertTrue(medium.contains("float3 lightRadiance = max(worldPush.lightRadiance.xyz"));
        assertTrue(medium.contains("float lightEnergy = dot(lightRadiance, float3(0.2126, 0.7152, 0.0722));"),
                "source hue should be reduced to a scalar before applying the blue haze colour");
        assertTrue(medium.contains("float3 hazeColor = lerp(float3(0.035, 0.095, 0.22),")
                        && medium.contains("float3 mediumSigma = max(waterExtinctionBase(worldPush.waterParams.xyz)"),
                "haze chromaticity must remain blue while its scalar attenuation follows the water medium");
        assertTrue(medium.contains("return hazeColor * lightEnergy * sourceAverage * hazeSigmaS * phase * viewIntegral;"),
                "normal water-path haze must remain tied to real light and should not glow independently");
        assertTrue(world.contains("float3 waterFogContribution = waterHazeScattering(rd, payload.hitT);"),
                "ordinary submerged path segments should accumulate haze before Beer-Lambert attenuation");
        assertFalse(world.contains("waterGodrayScattering") || world.contains("waterShaftNoise"),
                "the underwater godray feature should be fully removed");
    }

    @Test
    void underwaterTlasMissesConvergeToDenseBlueFarWaterInsteadOfBlack() throws IOException {
        Path root = repoRoot();
        String world = Files.readString(root.resolve("shaders/world/world.rgen.slang"));
        assertTrue(world.contains("float waterColumnDistance = skyWaterDistance;"),
                "unresolved underwater rays should use their full fallback distance, not only vertical camera depth");
        assertTrue(world.contains("float fallbackOpticalDepth = waterColumnDistance * 0.075 * hazeStrength;"));
        assertTrue(world.contains("float fallbackOpacity = 1.0 - exp(-max(fallbackOpticalDepth, 0.0));"),
                "the fallback should smoothly converge into denser haze with distance");
        assertTrue(world.contains("float3 farWaterBlue = lerp(float3(0.010, 0.035, 0.095),")
                        && world.contains("sky = waterColumnScatter + farWaterBlue * fallbackOpacity;"),
                "a missing underwater chunk should resolve to a dark-blue far-water environment rather than black");
        assertFalse(world.contains("waterColumnDistance = min(startDepth, skyWaterDistance)"),
                "the old depth-limited fallback leaves black gaps on long horizontal rays at chunk boundaries");
    }

    @Test
    void hazeIntegralStaysSmoothAndDoesNotUseCameraCentredPlaneIntersections() throws IOException {
        Path root = repoRoot();
        String medium = Files.readString(root.resolve("shaders/world/medium.slang"));
        String world = Files.readString(root.resolve("shaders/world/world.rgen.slang"));
        String miss = Files.readString(root.resolve("shaders/world/world.rmiss.slang"));
        assertTrue(medium.contains("float viewIntegralDirect = (1.0 - viewTransmittance) / max(sigmaT, 1.0e-5);"));
        assertTrue(medium.contains("float sourceOpticalDepth = sigmaT * (4.0 / lightPathY);"));
        assertTrue(medium.contains("float sourceAverage = exp(-max(sourceOpticalDepth, 0.0));"));
        assertFalse(medium.contains("sourceT0") || medium.contains("sourceT1")
                        || medium.contains("float startDepth"),
                "underwater haze density must not depend on a guessed camera-to-surface depth");
        assertFalse(world.contains("verticalGap / max(rd.y") || world.contains("skyWaterTransmittance")
                        || world.contains("surfaceVisibility") || world.contains("if (rd.y <= 0.015)"),
                "direction-dependent fake sky endpoints and ray-angle cutoffs create hard bands");
        assertFalse(miss.contains("underwaterSkyFallback"),
                "the underwater fallback belongs in the ray path and respects whether the segment is underwater");
        assertTrue(world.contains("float branchWeight = saturate(seg.prefixScatterWeight);"));
        assertTrue(world.contains("throughput *= mediumTransmittance;"));
    }

    @Test
    void underwaterHazeControlsRemainAvailableWithoutGodrayOptions() throws IOException {
        Path root = repoRoot();
        String screen = Files.readString(root.resolve("src/main/java/dev/comfyfluffy/caustica/client/gui/RtVideoOptionsScreen.java"));
        String subScreens = Files.readString(root.resolve("src/main/java/dev/comfyfluffy/caustica/client/gui/RtSubScreens.java"));
        String options = Files.readString(root.resolve("src/main/java/dev/comfyfluffy/caustica/client/RtVideoOptions.java"));
        String config = Files.readString(root.resolve("src/main/java/dev/comfyfluffy/caustica/CausticaConfig.java"));
        String composite = Files.readString(root.resolve("src/main/java/dev/comfyfluffy/caustica/rt/RtComposite.java"));
        assertTrue(screen.contains("RtSubScreens::underwaterSky"));
        assertTrue(subScreens.contains("UNDERWATER_SKY_DISTANCE"));
        assertTrue(options.contains("underwaterHazeEnabled()") && options.contains("underwaterBlueTint()"));
        assertTrue(options.contains("underwaterSkyOptions()"));
        assertFalse(composite.contains("underwaterSurfaceYRelative") || composite.contains("cachedUnderwaterSurface"),
                "a nearby overhead block must not be misclassified as the water surface for haze density");
        assertTrue(composite.contains("0.0f, // reserved: haze source attenuation no longer depends on a guessed local surface height"));
        assertTrue(config.contains("UNDERWATER_HAZE_STRENGTH") && config.contains("UNDERWATER_BLUE_TINT"));
        assertFalse(options.contains("underwaterGodrays") || config.contains("UNDERWATER_GODRAYS"));
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
