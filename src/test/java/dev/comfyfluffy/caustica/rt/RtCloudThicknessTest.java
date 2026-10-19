package dev.comfyfluffy.caustica.rt;

import dev.comfyfluffy.caustica.CausticaConfig;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The Min/Max Cloud Thickness contract.
 *
 * <p>These two sliders are the only player-facing control over how thick a generated cloud is, and the
 * requirement they carry is unusually concrete: 16..22 blocks, independently adjustable, always
 * ordered min &lt;= max, and — the part that is easy to lose in a refactor — actually used as the
 * geometric extent of the generated formations rather than as a density or opacity multiplier. A change
 * that quietly turned them into an opacity scale would still compile, still render, and still look
 * plausible in a screenshot, so the mapping is pinned here in two halves:
 *
 * <ul>
 *   <li><b>Configuration</b> — ranges, ordering and the repair path for out-of-order values arriving
 *       from a TOML file or a {@code -D} property.</li>
 *   <li><b>Shader wiring</b> — that {@code clouds.slang} reads the pair out of
 *       {@code cloudShape.xy}, lerps each formation's thickness between exactly those two numbers, and
 *       closes the volume at that height (so the value really is the extent of the density, not a scale
 *       on something else).</li>
 * </ul>
 */
final class RtCloudThicknessTest {
    private static final Path REPO_ROOT = repoRoot();
    private static final Path CLOUDS = REPO_ROOT.resolve("shaders/world/clouds.slang");
    private static final Path WORLD_COMMON = REPO_ROOT.resolve("shaders/world/world_common.slang");
    private static final Path RT_COMPOSITE =
            REPO_ROOT.resolve("src/main/java/dev/comfyfluffy/caustica/rt/RtComposite.java");

    private static final CausticaConfig.IntSetting MIN = CausticaConfig.Rt.Composite.CLOUD_MIN_THICKNESS;
    private static final CausticaConfig.IntSetting MAX = CausticaConfig.Rt.Composite.CLOUD_MAX_THICKNESS;

    @AfterEach
    void restoreDefaults() {
        MIN.resetToDefault();
        MAX.resetToDefault();
    }

    // ---------------------------------------------------------------- configuration

    @Test
    void sliderRangeIsSixteenToTwentyTwoBlocks() {
        assertEquals(16, CausticaConfig.Rt.Composite.CLOUD_MIN_THICKNESS_BLOCKS);
        assertEquals(22, CausticaConfig.Rt.Composite.CLOUD_MAX_THICKNESS_BLOCKS);

        CausticaConfig.Rt.Composite.setCloudMinThickness(4);
        assertEquals(16, MIN.value(), "the minimum must clamp to its declared lower bound");
        CausticaConfig.Rt.Composite.setCloudMaxThickness(99);
        assertEquals(22, MAX.value(), "the maximum must clamp to its declared upper bound");
    }

    @Test
    void defaultsSitInsideTheRangeAndRespectTheOrdering() {
        int min = MIN.defaultValue();
        int max = MAX.defaultValue();
        assertTrue(min >= 16 && min <= 22, "minimum default out of range: " + min);
        assertTrue(max >= 16 && max <= 22, "maximum default out of range: " + max);
        assertTrue(min <= max, "default thickness range is inverted: " + min + ".." + max);
    }

    @Test
    void raisingTheMinimumAboveTheMaximumPushesTheMaximum() {
        CausticaConfig.Rt.Composite.setCloudMinThickness(16);
        CausticaConfig.Rt.Composite.setCloudMaxThickness(18);
        CausticaConfig.Rt.Composite.setCloudMinThickness(21);

        assertEquals(21, MIN.value());
        assertEquals(21, MAX.value(), "the maximum must follow the minimum up instead of inverting");
    }

    @Test
    void loweringTheMaximumBelowTheMinimumPushesTheMinimumDown() {
        CausticaConfig.Rt.Composite.setCloudMinThickness(21);
        CausticaConfig.Rt.Composite.setCloudMaxThickness(22);
        CausticaConfig.Rt.Composite.setCloudMaxThickness(17);

        assertEquals(17, MAX.value());
        assertEquals(17, MIN.value(), "the minimum must follow the maximum down instead of inverting");
    }

    @Test
    void equalEndpointsAreAValidConfiguration() {
        // The requirement's explicit "min == max" case: every formation is generated at that one value.
        CausticaConfig.Rt.Composite.setCloudMinThickness(19);
        CausticaConfig.Rt.Composite.setCloudMaxThickness(19);
        assertEquals(19, CausticaConfig.Rt.Composite.cloudMinThicknessBlocks());
        assertEquals(19, CausticaConfig.Rt.Composite.cloudMaxThicknessBlocks());
    }

    @Test
    void anOutOfOrderPairFromTheConfigFileIsRepaired() {
        // A hand-edited caustica.toml can write any two numbers, and `set` only clamps each one
        // independently; the renderer must still see an ordered pair.
        MIN.set(22);
        MAX.set(17);
        assertEquals(17, CausticaConfig.Rt.Composite.cloudMinThicknessBlocks(),
                "reconciliation must pull the minimum down to the maximum");
        assertTrue(MIN.value() <= MAX.value(),
                "the pair handed to the shader is still inverted: " + MIN.value() + ".." + MAX.value());
    }

    @Test
    void bothEndpointsAreRegisteredForPageAndGlobalReset() {
        // "Reset to Defaults" on the clouds page iterates that page's own list; anything missing from it
        // would leave the player's value behind after a reset.
        List<CausticaConfig.RuntimeSetting<?>> page = CausticaConfig.settings().stream()
                .filter(setting -> setting == MIN || setting == MAX)
                .toList();
        assertEquals(2, page.size(), "the thickness settings must be registered runtime settings");
        for (CausticaConfig.RuntimeSetting<?> setting : page) {
            assertTrue(setting.key().startsWith("caustica.rt."),
                    setting.key() + " must use the caustica.rt.* property namespace");
            assertTrue(setting.tomlPath().startsWith("composite."),
                    setting.tomlPath() + " must persist under [composite] in caustica.toml");
        }
    }

    // ---------------------------------------------------------------- shader wiring

    /**
     * The generated thickness must be a LERP between the two slider lanes, i.e. the pair determines the
     * interval each formation's height is drawn from. Pinning the lerp (rather than just "the constants
     * are read somewhere") is what keeps a future edit from collapsing the two values into one, or from
     * substituting a density multiplier for the geometry.
     */
    @Test
    void formationThicknessLerpsBetweenTheTwoSliderLanes() throws IOException {
        String source = Files.readString(CLOUDS);
        String body = slice(source, "float cloudFormationThickness(", "float cloudFormationBase(");

        assertTrue(body.contains("push.cloudShape.x"),
                "the minimum thickness must come from cloudShape.x");
        assertTrue(body.contains("push.cloudShape.y"),
                "the maximum thickness must come from cloudShape.y");
        assertTrue(body.contains("max(push.cloudShape.y, minThickness)"),
                "the maximum must be floored at the minimum, so the lerp can never run backwards");
        assertTrue(body.contains("lerp(minThickness, maxThickness, t)"),
                "formation thickness must be lerped between the two sliders");
    }

    /**
     * The value must close the VOLUME: outside the formation's own [base, base + thickness] the density
     * is exactly zero. That is what makes the sliders geometry controls — and with min == max they make
     * every formation exactly that thick, which is the equality case the feature asks for.
     */
    @Test
    void formationThicknessClosesTheDensityVolume() throws IOException {
        String source = Files.readString(CLOUDS);
        String body = slice(source, "float cloudVolumeDensity(", "float cloudLightOpticalDepth(");

        assertTrue(body.contains("float rel = yInDeck - base;"),
                "the density must measure height above the formation's own base");
        assertTrue(body.contains("if (rel <= 0.0 || rel >= thickness)"),
                "the density must be zero outside the formation's own thickness");
        assertTrue(body.contains("float heightFraction = rel / thickness;"),
                "the vertical profile must be normalised by the formation's own thickness");
    }

    /** The slab the march integrates has to cover the thickest formation plus its base jitter. */
    @Test
    void marchSlabCoversTheThickestFormation() throws IOException {
        String composite = Files.readString(RT_COMPOSITE);
        assertTrue(composite.contains("thickness = maxThickness + CLOUD_BASE_JITTER_BLOCKS;"),
                "the volumetric slab must be the maximum formation thickness plus its base offset");
        assertTrue(composite.contains(
                        "new Float4(minThickness, maxThickness, cloudMorphOffsetBlocks(), 0f)"),
                "both slider values must be pushed in the cloudShape lane");
        assertTrue(Files.readString(WORLD_COMMON).contains("public float4   cloudShape;"),
                "WorldPush must carry the cloudShape lane");
    }

    /** Distance-faded detail must not turn into a second, distance-dependent thickness. */
    @Test
    void thicknessIsIndependentOfViewDistance() throws IOException {
        String source = Files.readString(CLOUDS);
        String body = slice(source, "float cloudFormationThickness(", "float cloudVolumeDensity(");
        assertFalse(body.contains("cloudAnchor.w"),
                "formation thickness must not depend on the view limit");
        assertFalse(body.contains("smoothstep(CLOUD_DETAIL_LOD"),
                "formation thickness must not depend on the detail LOD range");
    }

    // ---------------------------------------------------------------- helpers

    private static String slice(String source, String startNeedle, String endNeedle) {
        int start = source.indexOf(startNeedle);
        assertTrue(start >= 0, "missing shader snippet start: " + startNeedle);
        int end = source.indexOf(endNeedle, start);
        assertTrue(end > start, "missing shader snippet end: " + endNeedle);
        return source.substring(start, end);
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
