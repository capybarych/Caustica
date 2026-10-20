package dev.comfyfluffy.caustica.rt;

import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertTrue;

/** Regression guard: lack of NGX must disable DLSS-RR, not the independent Vulkan path tracer. */
final class RtOptionalDlssFallbackTest {
    @Test
    void failedNgxProbeDisablesOnlyRayReconstruction() throws IOException {
        Path root = repoRoot();
        String rr = Files.readString(root.resolve(
                "src/main/java/dev/comfyfluffy/caustica/rt/pipeline/RtDlssRr.java"));
        String composite = Files.readString(root.resolve(
                "src/main/java/dev/comfyfluffy/caustica/rt/RtComposite.java"));
        assertTrue(rr.contains("&& !INSTANCE.failed"),
                "RR's effective switch must respect the native backend failure latch");
        assertTrue(rr.contains("DLSS-RR unavailable; continuing with full-resolution path tracing and built-in denoising"));
        assertTrue(rr.contains("failed = true;") && rr.contains("return null;"),
                "an NGX probe failure must turn off RR and return a normal fallback result");
        assertTrue(composite.contains("queryOptimalRenderSize(width, height)"));
        assertTrue(composite.contains("svgfEnabled = !nrdEnabled && CausticaConfig.Rt.Denoise.ENABLED.value();"),
                "SVGF resources must be allocated when optional RR is unavailable");
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
