package dev.comfyfluffy.caustica.rt;

import dev.comfyfluffy.caustica.CausticaConfig;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Guards the DLSS Ray Reconstruction model-preset slider.
 *
 * <p>The slider edits one integer that is handed to NGX as
 * {@code NVSDK_NGX_RayReconstruction_Hint_Render_Preset}. Two things can silently break it: an
 * out-of-enum value (the DLL quietly falls back to its default while the UI claims a model) and a
 * preset change that does not recreate the RR feature (presets are captured at creation time, so the
 * image would keep the old model). Both are pinned here, together with the translation keys the
 * slider's value labels resolve through.
 */
final class RtDlssPresetTest {
    private static final Path REPO_ROOT = repoRoot();
    private static final Path RT_VIDEO_OPTIONS =
            REPO_ROOT.resolve("src/main/java/dev/comfyfluffy/caustica/client/RtVideoOptions.java");
    private static final Path RT_DLSS_RR =
            REPO_ROOT.resolve("src/main/java/dev/comfyfluffy/caustica/rt/pipeline/RtDlssRr.java");
    private static final Path EN_US =
            REPO_ROOT.resolve("src/main/resources/assets/caustica/lang/en_us.json");

    @Test
    void sliderOffersEveryUsableRayReconstructionPresetInOrder() {
        assertEquals(List.of(0, 4, 5, 6), CausticaConfig.Rt.DlssRr.PRESET_VALUES,
                "0 = DLL default, 4 = D, 5 = E, 6 = F; A/B/C were removed by NVIDIA and G+ reverts "
                        + "to the default behavior");
    }

    @Test
    void unknownStoredPresetsSnapToTheDefaultInsteadOfReachingNgx() {
        CausticaConfig.IntSetting preset = CausticaConfig.Rt.DlssRr.PRESET;
        int original = preset.value();
        try {
            preset.set(1); // preset A: removed from the enum, must never be pushed to NGX
            assertEquals(0, preset.value(), "a removed preset must snap to the DLL default");
            preset.set(7); // preset G: "reverts to default behavior"
            assertEquals(0, preset.value(), "presets past F must snap to the DLL default");
            for (int supported : CausticaConfig.Rt.DlssRr.PRESET_VALUES) {
                preset.set(supported);
                assertEquals(supported, preset.value(), "every offered preset must survive set()");
            }
        } finally {
            preset.set(original);
        }
    }

    @Test
    void sliderIsBuiltFromTheSettingAndItsValueLabelsExist() throws IOException {
        String options = Files.readString(RT_VIDEO_OPTIONS);
        assertTrue(options.contains("private static OptionInstance<Integer> dlssPreset()"),
                "the preset row must stay an integer OptionInstance (rendered as a slider)");
        assertTrue(options.contains("new OptionInstance.IntRange(0, DLSS_PRESET_ORDER.size() - 1)"),
                "the preset row must be a slider, not a cycle button");
        assertTrue(options.contains("position -> setting.set(DLSS_PRESET_ORDER.get(position))"),
                "moving the slider must write the NGX preset value, not the slider position");
        assertTrue(options.contains("DLSS_PRESET_ORDER = CausticaConfig.Rt.DlssRr.PRESET_VALUES"),
                "the slider order must come from the setting's own value list, or the two can drift");

        String lang = Files.readString(EN_US);
        for (int preset : CausticaConfig.Rt.DlssRr.PRESET_VALUES) {
            assertTrue(lang.contains("\"caustica.options.rt.dlssPreset." + preset + "\""),
                    "missing value label for preset " + preset + " (the slider would show a raw key)");
        }
        assertFalse(lang.contains("\"caustica.options.rt.dlssPreset.1\""),
                "preset A is gone from the enum; no label may resurrect it");
    }

    @Test
    void changingThePresetRecreatesTheFeature() throws IOException {
        String rr = Files.readString(RT_DLSS_RR);
        assertTrue(rr.contains("featurePreset != preset"),
                "ensureFeature must notice a preset change and recreate the feature — presets are "
                        + "captured at creation time, so otherwise the slider would do nothing");
        assertTrue(rr.contains("lib.createDlssd(cmd, renderWidth, renderHeight, displayWidth, displayHeight,"),
                "the selected preset must reach the shim's createDlssd");
        assertTrue(rr.contains("quality, FEATURE_FLAGS, preset)"),
                "the NGX call must pass the preset value unchanged");
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
