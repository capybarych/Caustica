package dev.comfyfluffy.caustica.rt;

import org.junit.jupiter.api.Test;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Keeps the configurable rain-lightning signal connected from Java into visible and diffuse RT paths. */
final class RtLightningShaderRegressionTest {
    @Test
    void lightningHasAConfigurableRandomScheduleAndPushLane() throws IOException {
        Path root = repoRoot();
        String config = Files.readString(root.resolve("src/main/java/dev/comfyfluffy/caustica/CausticaConfig.java"));
        String controller = Files.readString(root.resolve("src/main/java/dev/comfyfluffy/caustica/rt/RtLightning.java"));
        String common = Files.readString(root.resolve("shaders/world/world_common.slang"));
        String composite = Files.readString(root.resolve("src/main/java/dev/comfyfluffy/caustica/rt/RtComposite.java"));
        assertTrue(config.contains("LIGHTNING_STRIKES_PER_MINUTE"));
        assertTrue(controller.contains("nextDelay(strikesPerMinute)"));
        assertTrue(controller.contains("60.0 / rate"),
                "lightning rate must be converted from strikes per minute to a mean interval");
        assertTrue(controller.contains("Heightmap.Types.MOTION_BLOCKING_NO_LEAVES"));
        assertTrue(common.contains("public float4   lightning;"));
        assertTrue(composite.contains("CausticaConfig.Rt.Composite.LIGHTNING_STRIKES_PER_MINUTE.value()"));
    }

    @Test
    void lightningPlaysConfigurablePositionalThunderOncePerStrike() throws IOException {
        Path root = repoRoot();
        String config = Files.readString(root.resolve("src/main/java/dev/comfyfluffy/caustica/CausticaConfig.java"));
        String controller = Files.readString(root.resolve("src/main/java/dev/comfyfluffy/caustica/rt/RtLightning.java"));
        String options = Files.readString(root.resolve("src/main/java/dev/comfyfluffy/caustica/client/RtVideoOptions.java"));
        assertTrue(config.contains("LIGHTNING_SOUND_ENABLED"));
        assertTrue(config.contains("LIGHTNING_SOUND_VOLUME"));
        assertTrue(controller.contains("SoundEvents.LIGHTNING_BOLT_THUNDER"));
        assertTrue(controller.contains("level.playLocalSound(strikeX, strikeY, strikeZ"));
        assertTrue(controller.contains("SoundSource.WEATHER, volume, pitch, true"),
                "thunder should be positional and use distance delay");
        assertTrue(options.contains("lightningSoundEnabled()") && options.contains("lightningSoundVolume()"));
        assertTrue(options.contains("lightningStrikesPerMinute()"));
        assertTrue(options.contains("strikes/min"), "the menu should display a rate, not seconds");
    }

    @Test
    void lightningIlluminatesTerrainAndDrawsAProceduralBoltDuringTheFlash() throws IOException {
        Path root = repoRoot();
        String lighting = Files.readString(root.resolve("shaders/world/lighting.slang"));
        String raygen = Files.readString(root.resolve("shaders/world/world.rgen.slang"));
        String miss = Files.readString(root.resolve("shaders/world/world.rmiss.slang"));
        assertTrue(lighting.contains("public float3 lightningIrradiance("));
        assertTrue(raygen.contains("+ lightningIrradiance("));
        assertTrue(miss.contains("lightningBoltCoverage(WorldRayOrigin(), dir, worldPush.lightning.xyz)"));
        assertTrue(miss.contains("col += float3(2.5, 3.2, 4.8) * flash"));
        assertFalse(miss.contains("abs(sin(phase))"));
    }

    @Test
    void underwaterGodraysAreRemovedAndLegacyGlowScatteringRemainsAbsent() throws IOException {
        Path root = repoRoot();
        String options = Files.readString(root.resolve("src/main/java/dev/comfyfluffy/caustica/client/RtVideoOptions.java"));
        String config = Files.readString(root.resolve("src/main/java/dev/comfyfluffy/caustica/CausticaConfig.java"));
        String common = Files.readString(root.resolve("shaders/world/world_common.slang"));
        String world = Files.readString(root.resolve("shaders/world/world.rgen.slang"));
        String medium = Files.readString(root.resolve("shaders/world/medium.slang"));
        assertFalse(options.contains("underwaterGodrays"), "underwater godray options should be removed from the UI");
        assertFalse(config.contains("UNDERWATER_GODRAYS"), "underwater godray config should be removed");
        assertFalse(common.contains("underwaterGodray"), "underwater godray push lanes should be removed");
        assertFalse(world.contains("waterGodrayScattering") || world.contains("waterShaftNoise"),
                "underwater godray shader code should be removed");
        assertFalse(medium.contains("waterSegmentInscatter"),
                "legacy glow scattering must remain absent from the underwater medium shader");
    }

    private static Path repoRoot() {
        Path dir = Path.of("").toAbsolutePath();
        for (Path candidate = dir; candidate != null; candidate = candidate.getParent()) {
            if (Files.isDirectory(candidate.resolve("shaders/world"))
                    && Files.isDirectory(candidate.resolve("src/main/java"))) return candidate;
        }
        throw new IllegalStateException("could not locate repository root from " + dir);
    }
}
