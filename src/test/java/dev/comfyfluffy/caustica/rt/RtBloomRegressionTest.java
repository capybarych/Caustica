package dev.comfyfluffy.caustica.rt;

import dev.comfyfluffy.caustica.CausticaConfig;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Regression tests for the screen-space multi-scale HDR Bloom pipeline, shaders, configuration,
 * and RT Video Settings UI integration.
 */
final class RtBloomRegressionTest {
    private static final Path REPO_ROOT = repoRoot();

    @Test
    void bloomSettingsHaveSensibleDefaultsAndClampRanges() {
        CausticaConfig.ensureRegistered();

        CausticaConfig.BooleanSetting enabled = CausticaConfig.Rt.Bloom.ENABLED;
        CausticaConfig.StringSetting quality = CausticaConfig.Rt.Bloom.QUALITY;
        CausticaConfig.FloatSetting intensity = CausticaConfig.Rt.Bloom.INTENSITY;
        CausticaConfig.FloatSetting threshold = CausticaConfig.Rt.Bloom.THRESHOLD;
        CausticaConfig.FloatSetting softKnee = CausticaConfig.Rt.Bloom.SOFT_KNEE;
        CausticaConfig.FloatSetting radius = CausticaConfig.Rt.Bloom.RADIUS;
        CausticaConfig.FloatSetting colorInfluence = CausticaConfig.Rt.Bloom.COLOR_INFLUENCE;

        try {
            enabled.resetToDefault();
            quality.resetToDefault();
            intensity.resetToDefault();
            threshold.resetToDefault();
            softKnee.resetToDefault();
            radius.resetToDefault();
            colorInfluence.resetToDefault();

            assertEquals(Boolean.TRUE, enabled.value());
            assertEquals("high", quality.get());
            assertEquals(0.15f, intensity.value(), 1.0e-6f);
            assertEquals(1.0f, threshold.value(), 1.0e-6f);
            assertEquals(0.5f, softKnee.value(), 1.0e-6f);
            assertEquals(0.65f, radius.value(), 1.0e-6f);
            assertEquals(1.0f, colorInfluence.value(), 1.0e-6f);

            // Clamping behavior
            intensity.set(-1.0f);
            assertEquals(0.0f, intensity.value(), 1.0e-6f);
            intensity.set(5.0f);
            assertEquals(1.0f, intensity.value(), 1.0e-6f);

            threshold.set(-2.0f);
            assertEquals(0.0f, threshold.value(), 1.0e-6f);
            threshold.set(25.0f);
            assertEquals(10.0f, threshold.value(), 1.0e-6f);

            softKnee.set(-0.5f);
            assertEquals(0.0f, softKnee.value(), 1.0e-6f);
            softKnee.set(2.0f);
            assertEquals(1.0f, softKnee.value(), 1.0e-6f);

            radius.set(-0.2f);
            assertEquals(0.0f, radius.value(), 1.0e-6f);
            radius.set(1.5f);
            assertEquals(1.0f, radius.value(), 1.0e-6f);

            colorInfluence.set(-1.0f);
            assertEquals(0.0f, colorInfluence.value(), 1.0e-6f);
            colorInfluence.set(3.0f);
            assertEquals(2.0f, colorInfluence.value(), 1.0e-6f);
        } finally {
            enabled.resetToDefault();
            quality.resetToDefault();
            intensity.resetToDefault();
            threshold.resetToDefault();
            softKnee.resetToDefault();
            radius.resetToDefault();
            colorInfluence.resetToDefault();
        }
    }

    @Test
    void bloomQualityPresetsProduceDistinctPyramidDepthsAndIndices() {
        CausticaConfig.StringSetting quality = CausticaConfig.Rt.Bloom.QUALITY;
        try {
            quality.set("low");
            assertEquals("low", quality.get());
            assertEquals(0, CausticaConfig.Rt.Bloom.qualityIndex());
            assertEquals(3, CausticaConfig.Rt.Bloom.mipCountForQuality(0));

            quality.set("medium");
            assertEquals("medium", quality.get());
            assertEquals(1, CausticaConfig.Rt.Bloom.qualityIndex());
            assertEquals(4, CausticaConfig.Rt.Bloom.mipCountForQuality(1));

            quality.set("high");
            assertEquals("high", quality.get());
            assertEquals(2, CausticaConfig.Rt.Bloom.qualityIndex());
            assertEquals(5, CausticaConfig.Rt.Bloom.mipCountForQuality(2));

            quality.set("ultra");
            assertEquals("ultra", quality.get());
            assertEquals(3, CausticaConfig.Rt.Bloom.qualityIndex());
            assertEquals(6, CausticaConfig.Rt.Bloom.mipCountForQuality(3));

            quality.set("unknown-quality");
            assertEquals("high", quality.get(), "invalid quality strings must fall back to high");
        } finally {
            quality.resetToDefault();
        }
    }

    @Test
    void bloomShadersAndPipelineKeepMatchingLayoutsAndMultiDispatchSets() throws IOException {
        String downsample = Files.readString(REPO_ROOT.resolve("shaders/display/bloom_downsample.comp"));
        String upsample = Files.readString(REPO_ROOT.resolve("shaders/display/bloom_upsample.comp"));
        String composite = Files.readString(REPO_ROOT.resolve("shaders/display/bloom_composite.comp"));
        String pipeline = Files.readString(
                REPO_ROOT.resolve("src/main/java/dev/comfyfluffy/caustica/rt/pipeline/RtBloomPipeline.java"));

        assertTrue(downsample.contains("layout(binding = 0) uniform sampler2D srcTex;"));
        assertTrue(downsample.contains("layout(binding = 1, r32f) readonly uniform image2D exposureImage;"));
        assertTrue(downsample.contains("layout(binding = 2, rgba16f) writeonly uniform image2D dstImage;"));
        assertTrue(downsample.contains("karisWeight"), "downsample shader must include Karis anti-firefly weighting");

        assertTrue(upsample.contains("layout(binding = 0) uniform sampler2D coarseTex;"));
        assertTrue(upsample.contains("layout(binding = 1, rgba16f) uniform image2D fineImage;"));
        assertTrue(upsample.contains("mix(fine, upsampled, scatter)"),
                "upsample shader must use energy-conserving convex combination across mips");

        assertTrue(composite.contains("layout(binding = 0) uniform sampler2D bloomTex;"));
        assertTrue(composite.contains("layout(binding = 1, r32f) readonly uniform image2D exposureImage;"));
        assertTrue(composite.contains("layout(binding = 2, rgba16f) uniform image2D sceneImage;"));
        assertTrue(composite.contains("pc.colorInfluence"), "composite shader must apply colorInfluence");

        assertTrue(pipeline.contains("downsampleSets = new long[MAX_MIPS]"),
                "each downsample mip dispatch in a command buffer must own its own descriptor set");
        assertTrue(pipeline.contains("upsampleSets = new long[MAX_MIPS - 1]"),
                "each upsample mip dispatch in a command buffer must own its own descriptor set");
        assertTrue(pipeline.contains("public void invalidateBindings()"),
                "pipeline must invalidate cached image views when targets are recreated");
    }

    @Test
    void bloomRunsBetweenExposureAndDisplayToneMappingAndBypassesWhenDisabled() throws IOException {
        String rtComposite = Files.readString(
                REPO_ROOT.resolve("src/main/java/dev/comfyfluffy/caustica/rt/RtComposite.java"));

        int exposureIdx = rtComposite.indexOf("exposure.record(ctx, cmd, stack, rrOutput);");
        int bloomIdx = rtComposite.indexOf("bloomPipeline.record(cmd, stack, rrOutput, exposure.image());");
        int displayIdx = rtComposite.indexOf("displayPipeline.dispatch(cmd, displayW, displayH,");

        assertTrue(exposureIdx > 0, "RtComposite must record scene exposure");
        assertTrue(bloomIdx > exposureIdx,
                "Bloom must run after exposure metering so bloom cannot feed back into auto-exposure");
        assertTrue(displayIdx > bloomIdx,
                "Bloom must composite into linear HDR rrOutput before display tone mapping");
        assertTrue(rtComposite.contains("debugView == 0 && bloomPipeline != null"),
                "Bloom must be bypassed when a debug view is active");
        assertTrue(rtComposite.contains("CausticaConfig.Rt.Bloom.ENABLED.value()"),
                "RtComposite must gate bloom execution on the master enable toggle");
    }

    @Test
    void bloomSettingsAreWiredIntoUiAndLocalizations() throws IOException {
        String videoOptions = Files.readString(
                REPO_ROOT.resolve("src/main/java/dev/comfyfluffy/caustica/client/RtVideoOptions.java"));
        String subScreens = Files.readString(
                REPO_ROOT.resolve("src/main/java/dev/comfyfluffy/caustica/client/gui/RtSubScreens.java"));
        String hubScreen = Files.readString(
                REPO_ROOT.resolve("src/main/java/dev/comfyfluffy/caustica/client/gui/RtVideoOptionsScreen.java"));
        String enUs = Files.readString(
                REPO_ROOT.resolve("src/main/resources/assets/caustica/lang/en_us.json"));
        String ptBr = Files.readString(
                REPO_ROOT.resolve("src/main/resources/assets/caustica/lang/pt_br.json"));

        assertTrue(videoOptions.contains("public static OptionInstance<?>[] bloomOptions()"));
        assertTrue(subScreens.contains("public static RtSettingsSubScreen bloom(Screen parent)"));
        assertTrue(subScreens.contains("bloomSettings()"));
        assertTrue(hubScreen.contains("RtSubScreens.openButton(this, \"caustica.options.rt.bloomHeader\", RtSubScreens::bloom)"));

        List<String> requiredKeys = List.of(
                "caustica.options.rt.bloomHeader",
                "caustica.options.rt.bloom",
                "caustica.options.rt.bloom.tooltip",
                "caustica.options.rt.bloomQuality",
                "caustica.options.rt.bloomQuality.tooltip",
                "caustica.options.rt.bloomQuality.low",
                "caustica.options.rt.bloomQuality.medium",
                "caustica.options.rt.bloomQuality.high",
                "caustica.options.rt.bloomQuality.ultra",
                "caustica.options.rt.bloomIntensity",
                "caustica.options.rt.bloomIntensity.tooltip",
                "caustica.options.rt.bloomThreshold",
                "caustica.options.rt.bloomThreshold.tooltip",
                "caustica.options.rt.bloomSoftKnee",
                "caustica.options.rt.bloomSoftKnee.tooltip",
                "caustica.options.rt.bloomRadius",
                "caustica.options.rt.bloomRadius.tooltip",
                "caustica.options.rt.bloomColorInfluence",
                "caustica.options.rt.bloomColorInfluence.tooltip"
        );
        for (String key : requiredKeys) {
            assertTrue(enUs.contains("\"" + key + "\""), "Missing en_us translation key: " + key);
            assertTrue(ptBr.contains("\"" + key + "\""), "Missing pt_br translation key: " + key);
        }
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
