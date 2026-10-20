package dev.comfyfluffy.caustica.rt.entity;

import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertTrue;

/** Regression checks for the small vanilla rain-impact/splash particles as well as weather columns. */
final class RtRainSplashAppearanceTest {
    @Test
    void rainImpactParticlesUseNeutralRefractiveWeatherPath() throws IOException {
        Path root = repoRoot();
        String capture = Files.readString(root.resolve("src/main/java/dev/comfyfluffy/caustica/rt/entity/RtEntities.java"));
        String hit = Files.readString(root.resolve("shaders/world/world.rchit.slang"));
        String primary = Files.readString(root.resolve("shaders/world/world_primary.rgen.slang"));
        String raygen = Files.readString(root.resolve("shaders/world/world.rgen.slang"));
        assertTrue(capture.contains("isRainImpactParticle(p) ? RtEntityCapture.PRIM_FLAG_WEATHER : 0"));
        assertTrue(capture.contains("name.contains(\"rainparticle\")"));
        assertTrue(hit.contains("? half3(0.90h, 0.94h, 0.97h)"),
                "weather and splash sprites should not inherit vanilla blue RGB tint");
        assertTrue(primary.contains("? float3(0.90, 0.94, 0.97) : float3(payload.albedo)"),
                "primary guides should use the same neutral weather albedo");
        assertTrue(raygen.contains("float3 dropTint = float3(0.92, 0.96, 0.98)"));
        assertTrue(raygen.contains("float weatherSheet = lerp(0.0025, 0.018"),
                "drop glints should remain translucent/subtle rather than opaque blue pixels");
    }

    private static Path repoRoot() {
        Path dir = Path.of("").toAbsolutePath();
        for (Path candidate = dir; candidate != null; candidate = candidate.getParent()) {
            if (Files.isDirectory(candidate.resolve("shaders/world"))
                    && Files.isDirectory(candidate.resolve("src/main/java"))) {
                return candidate;
            }
        }
        throw new IllegalStateException("could not locate repository root");
    }
}
