package dev.comfyfluffy.caustica;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;

import com.electronwill.nightconfig.core.CommentedConfig;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Guards the semantics the settings UI's "Reset to Defaults" buttons are built on.
 *
 * <p>Every Caustica sub-screen starts with a per-page reset row and the hub with a global one, all of
 * them implemented as {@link CausticaConfig.RuntimeSetting#resetToDefault()} over a settings list. If a
 * future setting quietly stopped snapping back to its factory value — especially an angle setting whose
 * {@code set()} consumes degrees while the stored default is already radians — every Reset button would
 * lie to the player, so the contract is pinned here for every registered setting.
 */
final class RtConfigDefaultsTest {

    @Test
    void settingsRegistryIsPopulated() {
        assertFalse(CausticaConfig.settings().isEmpty(),
                "the settings registry must not be empty: the UI resets iterate it");
    }

    @Test
    void factoryDefaultsMatchTheShippedPreset() throws IOException {
        assertEquals(4, CausticaConfig.Rt.WORKER_THREADS.defaultValue().intValue());
        assertEquals(8, CausticaConfig.Rt.Composite.MAX_BOUNCES.defaultValue().intValue());
        assertEquals("volumetric", CausticaConfig.Rt.Composite.CLOUD_STYLE.defaultValue());
        assertTrue(CausticaConfig.Rt.Composite.FOG_ENABLED.defaultValue());
        assertEquals(0.1f, CausticaConfig.Rt.Composite.UNDERWATER_HAZE_STRENGTH.defaultValue(), 1.0e-6f);
        assertEquals(0.5f, CausticaConfig.Rt.Composite.UNDERWATER_BLUE_TINT.defaultValue(), 1.0e-6f);
        assertEquals(6.0f, CausticaConfig.Rt.Composite.WETNESS_TRANSITION_DISTANCE.defaultValue(), 1.0e-6f);
        assertEquals(2.0f, CausticaConfig.Rt.Composite.LIGHTNING_STRIKES_PER_MINUTE.defaultValue(), 1.0e-6f);
        assertTrue(CausticaConfig.Rt.Sharc.ENABLED.defaultValue());
        assertEquals(6, CausticaConfig.Rt.DlssRr.PRESET.defaultValue().intValue());
        assertEquals("psychov", CausticaConfig.Rt.Tonemapping.OPERATOR.defaultValue());

        try (InputStream stream = CausticaConfig.class.getResourceAsStream("/caustica-defaults.toml")) {
            assertNotNull(stream, "the bundled first-launch preset must be packaged as a resource");
            String preset = new String(stream.readAllBytes(), StandardCharsets.UTF_8);
            assertTrue(preset.contains("config-version = 1"));
            assertTrue(preset.contains("underwater-haze-strength = 0.1"));
            assertTrue(preset.contains("wetness-transition-distance = 6.0"));
            assertFalse(preset.toLowerCase(java.util.Locale.ROOT).contains("godray"),
                    "the bundled preset must not reintroduce removed underwater godray settings");
        }
    }

    @Test
    void existingFactoryDefaultsMigrateButCustomValuesArePreserved() {
        CommentedConfig config = CommentedConfig.inMemory();
        config.set("config-version", 0);
        config.set("composite.underwater-haze-strength", 1.0);
        config.set("composite.cloud-style", "classic");
        config.set("composite.cloud-height", 420.0); // A player's customization must survive.
        config.set("composite.cloud-coverage", 0.55);

        int changed = CausticaConfig.migrateLegacyFactoryDefaults(config);

        assertTrue(changed >= 3);
        assertEquals(1, ((Number) config.get("config-version")).intValue());
        assertEquals(0.1, ((Number) config.get("composite.underwater-haze-strength")).doubleValue(), 1.0e-6);
        assertEquals("volumetric", config.get("composite.cloud-style"));
        assertEquals(420.0, ((Number) config.get("composite.cloud-height")).doubleValue(), 1.0e-6);
        assertEquals(0.4, ((Number) config.get("composite.cloud-coverage")).doubleValue(), 1.0e-6);

        // Versioning makes the migration one-shot; intentionally selecting a legacy value later is respected.
        config.set("composite.underwater-haze-strength", 1.0);
        assertEquals(0, CausticaConfig.migrateLegacyFactoryDefaults(config));
        assertEquals(1.0, ((Number) config.get("composite.underwater-haze-strength")).doubleValue(), 1.0e-6);
    }

    @Test
    void everySettingRestoresItsExactDefaultAfterAChange() {
        for (CausticaConfig.RuntimeSetting<?> setting : CausticaConfig.settings()) {
            Object factoryDefault = setting.defaultValue();
            perturb(setting);
            setting.resetToDefault();
            assertEquals(factoryDefault, setting.get(),
                    setting.key() + " did not snap back to its factory default after a change");
        }
    }

    @Test
    void radiansSettingsResetWithoutReapplyingTheInputTransform() {
        // Sun angular radius: stored in radians, but set() consumes degrees. A naive
        // set(defaultValue()) would convert the radian default a second time and pin the sun at
        // ~0.01 degrees — exactly the trap the custom resetToDefault() exists to avoid.
        CausticaConfig.FloatSetting sun = CausticaConfig.Rt.Composite.SUN_ANGULAR_RADIUS;
        // Tests must compare against the factory default, not the current value: another test or a
        // user's existing config may have changed the live setting before this test runs.
        float def = sun.defaultValue();
        sun.resetToDefault();
        sun.set(5.0f);
        assertNotEquals(def, sun.value(), 1.0e-9f, "perturbing the slider must change the value");
        sun.resetToDefault();
        assertEquals(def, sun.value(), "reset must restore the stored default verbatim");
        sun.set(0.6f); // the stock value written back through the public degrees path
        assertEquals(def, sun.value(), 1.0e-7f,
                "writing the stock degrees through set() must reproduce the stored radians");
    }

    @SuppressWarnings("unchecked")
    private static <T> void perturb(CausticaConfig.RuntimeSetting<T> setting) {
        T factoryDefault = setting.defaultValue();
        if (factoryDefault instanceof Boolean b) {
            ((CausticaConfig.RuntimeSetting<Boolean>) setting).set(!b);
        } else if (factoryDefault instanceof Integer i) {
            ((CausticaConfig.RuntimeSetting<Integer>) setting).set(i + 1);
        } else if (factoryDefault instanceof Float f) {
            ((CausticaConfig.RuntimeSetting<Float>) setting).set(f + 0.5f);
        } else {
            // String mode pickers (exposure/cloud style/tonemap) and the optional NGX path setting,
            // whose default is null. Sanitizers may map the junk value anywhere; the reset must still
            // land on the exact default.
            ((CausticaConfig.RuntimeSetting<String>) setting).set("perturbed-test-value");
        }
    }
}
