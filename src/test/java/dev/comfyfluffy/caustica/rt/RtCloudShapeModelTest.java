package dev.comfyfluffy.caustica.rt;

import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Guards the invariants the volumetric shape model is built on.
 *
 * <p>These are not style preferences — each one, if broken, produces a specific visible defect that
 * compiles and runs perfectly happily:
 *
 * <ul>
 *   <li><b>Power-of-two divisors and phase offsets.</b> The whole field is exactly periodic, and the
 *       renderer pushes a sample anchor already reduced modulo the coarsest lattice. A non-power-of-two
 *       divisor stops the lattices sharing that period, so the deck visibly snaps to a different
 *       pattern when the wind scroll crosses the anchor wrap.</li>
 *   <li><b>The churn period covering every Y lattice.</b> The CPU wraps the shape-evolution offset to
 *       one period. If a lattice no longer divides that period, the wrap becomes a discontinuity and
 *       the whole sky jumps whenever the offset rolls over.</li>
 *   <li><b>The crown envelope actually closing.</b> The isosurface threshold is raised toward
 *       {@code CLOUD_ISO_TOP} at the top of a formation. If that value is not above what coverage,
 *       the belly bulge and the noise can already produce, the mass never closes and every cloud ends
 *       at the formation's own ceiling — a flat-topped deck, i.e. the layered look this rework
 *       removed.</li>
 * </ul>
 */
final class RtCloudShapeModelTest {
    private static final Path REPO_ROOT = repoRoot();
    private static final Path CLOUDS = REPO_ROOT.resolve("shaders/world/clouds.slang");
    private static final Path RT_COMPOSITE =
            REPO_ROOT.resolve("src/main/java/dev/comfyfluffy/caustica/rt/RtComposite.java");

    @Test
    void everyShapeLatticeIsAPowerOfTwoDivisor() throws IOException {
        Map<String, Double> c = floats(Files.readString(CLOUDS));
        for (String name : new String[] {
                "CLOUD_WEATHER_DIV", "CLOUD_FORMATION_DIV", "CLOUD_WARP_DIV",
                "CLOUD_SHAPE_DIV", "CLOUD_LOBE_DIV", "CLOUD_DETAIL_DIV"}) {
            double divisor = c.get(name);
            assertTrue(divisor > 0.0, name + " must be positive");
            assertTrue(isPowerOfTwo(divisor),
                    name + " = " + divisor + " is not a power of two; the anchor wrap would no longer be"
                            + " a whole period in this lattice and the deck would snap on rollover");
        }
    }

    /**
     * The Y anisotropy values exist so a 16..22-block formation is a couple of lobes tall rather than
     * one lobe tall (a wall). They also move the lattices' Y periods, which must still divide the CPU's
     * churn wrap.
     */
    @Test
    void churnPeriodIsAWholePeriodOfEveryDisplacedLattice() throws IOException {
        Map<String, Double> c = floats(Files.readString(CLOUDS));
        double cells = c.get("CLOUD_CELL_BLOCKS");
        double periodCells = c.get("CLOUD_PERIOD_CELLS");
        double morphPeriod = c.get("CLOUD_MORPH_PERIOD_BLOCKS");

        assertTrue(morphPeriod > 0.0, "CLOUD_MORPH_PERIOD_BLOCKS must be positive");
        for (String[] lattice : new String[][] {
                {"CLOUD_SHAPE_DIV", "CLOUD_SHAPE_Y_STRETCH"},
                {"CLOUD_LOBE_DIV", "CLOUD_LOBE_Y_STRETCH"},
                {"CLOUD_DETAIL_DIV", null}}) {
            double yPeriod = cells * periodCells * c.get(lattice[0]);
            if (lattice[1] != null) {
                yPeriod /= c.get(lattice[1]);
            }
            assertTrue(divides(morphPeriod, yPeriod),
                    lattice[0] + "'s Y lattice period (" + yPeriod + " blocks) does not divide "
                            + "CLOUD_MORPH_PERIOD_BLOCKS (" + morphPeriod + "); the churn wrap would"
                            + " snap the cloudscape");
        }

        // The CPU wraps the offset to exactly this period; the two sides must agree.
        String composite = Files.readString(RT_COMPOSITE);
        var match = Pattern.compile(
                        "CLOUD_MORPH_PERIOD_BLOCKS\\s*=\\s*([\\d.]+)\\s*;")
                .matcher(composite);
        assertTrue(match.find(), "RtComposite must declare CLOUD_MORPH_PERIOD_BLOCKS");
        assertEquals(morphPeriod, Double.parseDouble(match.group(1)), 1.0e-6,
                "the churn period differs between clouds.slang and RtComposite");
    }

    @Test
    void crownEnvelopeClosesTheFormationInsteadOfCuttingIt() throws IOException {
        Map<String, Double> c = floats(Files.readString(CLOUDS));
        double crownStart = c.get("CLOUD_CROWN_START");
        double crownVariance = c.get("CLOUD_CROWN_VARIANCE");
        double isoTop = c.get("CLOUD_ISO_TOP");
        double isoBase = c.get("CLOUD_ISO_BASE");
        double isoCoverage = c.get("CLOUD_ISO_COVERAGE");
        double bulge = c.get("CLOUD_BULGE");

        assertTrue(crownStart > 0.25 && crownStart < 0.95,
                "CLOUD_CROWN_START = " + crownStart + " must leave room for a body below the crown");
        assertTrue(crownStart - crownVariance / 2.0 > 0.25,
                "the most aggressive crown variation would close the formation almost immediately");
        assertTrue(crownStart + crownVariance / 2.0 < 1.0,
                "the most relaxed crown variation would never close the formation");

        // The lowest threshold the noise can be reduced to, at full coverage and full belly bulge.
        double lowestIso = isoBase - isoCoverage - bulge;
        assertTrue(isoTop > lowestIso,
                "CLOUD_ISO_TOP = " + isoTop + " must exceed the lowest attainable threshold ("
                        + lowestIso + "), or the envelope cannot close a dense formation and every"
                        + " cloud would be sliced flat at its own ceiling");
        assertTrue(isoTop <= 1.0, "CLOUD_ISO_TOP above 1.0 would make the envelope fully opaque");
    }

    @Test
    void perlinWorleyBlendStaysAMix() throws IOException {
        Map<String, Double> c = floats(Files.readString(CLOUDS));
        double mix = c.get("CLOUD_WORLEY_MIX");
        double softness = c.get("CLOUD_SHAPE_SOFTNESS");
        double erosion = c.get("CLOUD_EROSION");
        assertTrue(mix >= 0.0 && mix <= 1.0, "CLOUD_WORLEY_MIX must be a lerp weight, was " + mix);
        assertTrue(softness > 0.0 && softness < 0.5,
                "the iso ramp must stay narrow enough to keep an edge, was " + softness);
        assertTrue(erosion >= 0.0 && erosion < 1.0,
                "the erosion strength must stay below 1 or it removes the whole shell, was " + erosion);
    }

    // ---------------------------------------------------------------- helpers

    /** All {@code public static const float NAME = value;} declarations, by name. */
    private static Map<String, Double> floats(String source) {
        Map<String, Double> values = new LinkedHashMap<>();
        Matcher m = Pattern.compile(
                "public static const float\\s+(\\w+)\\s*=\\s*(-?[\\d.]+)\\s*;").matcher(source);
        while (m.find()) {
            values.put(m.group(1), Double.parseDouble(m.group(2)));
        }
        // Integers that participate as proportions (CLOUD_PERIOD_CELLS) live in their own form.
        m = Pattern.compile("public static const int\\s+(\\w+)\\s*=\\s*(\\d+)\\s*;").matcher(source);
        while (m.find()) {
            values.putIfAbsent(m.group(1), Double.parseDouble(m.group(2)));
        }
        return values;
    }

    private static boolean isPowerOfTwo(double value) {
        double log = Math.log(value) / Math.log(2.0);
        return Math.abs(log - Math.rint(log)) < 1.0e-9 && value >= 1.0;
    }

    private static boolean divides(double whole, double part) {
        double ratio = whole / part;
        return Math.abs(ratio - Math.rint(ratio)) < 1.0e-9;
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
