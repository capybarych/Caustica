package dev.comfyfluffy.caustica.rt;

import java.util.Random;

/**
 * Host-side blue-noise dither tile for the path tracer's first stochastic sample.
 *
 * <h2>Why a texture instead of a hash</h2>
 * The tracer used to draw every random decision from a white-noise PCG stream. White noise leaves
 * the residual sampling error as per-pixel grain with energy spread evenly across all spatial
 * frequencies, which reads as harsh speckle and is the hardest input for a spatial denoiser. Blue
 * noise puts that energy at high spatial frequencies: neighbouring pixels get well-distributed
 * sample values instead of clumps of similar ones, so the same variance looks finer and the
 * denoiser's reconstruction filters can remove far more of it.
 *
 * <h2>What is generated</h2>
 * A 64x64 tile produced with Ulichney's void-and-cluster algorithm, four independent masks packed
 * one texel per 32-bit word (R/G = the BSDF direction pair, B/A = the NEE light-position pair). Each
 * 8-bit channel is a permutation of 0..255 repeated 16 times, so the values are exactly uniform
 * over the tile. {@code math.slang} samples it at pixel coordinates, masks with {@code TILE - 1}
 * (which is why the tile edge is a power of two), and applies a per-frame toroidal value shift so
 * one spatial tile also drives an unbiased temporal sequence.
 *
 * <h2>Generation cost</h2>
 * Four 64x64 void-and-cluster runs take well under a tenth of a second: each iteration only updates
 * the kernel footprint of the toggled texel (49 cells), and the full-tile scans are 4096 cells. The
 * result is generated once per session and cached; the renderer copies it into each BDA ring slot
 * when the ring is created.
 */
public final class RtBlueNoise {
    public static final RtBlueNoise INSTANCE = new RtBlueNoise();

    /** Tile edge in texels. Must stay a power of two: the shader masks pixel coordinates with it. */
    public static final int TILE = 64;
    private static final int TEXELS = TILE * TILE;

    /** One packed RGBA8 texel per word; see {@link #words()}. */
    public static final int WORDS = TEXELS;
    /** Upload footprint of the whole tile. */
    public static final int BYTES = WORDS * 4;

    // Void-and-cluster parameters. The classic (sigma 1.5, 10% initial ones) pair leaves too much
    // low-frequency energy for a dither mask at this size; a tighter kernel and a sparse initial
    // seed give the fine-grained spectrum the path tracer wants (measured lag-1 correlation around
    // -0.5, 2x2 block variance about a seventh of white noise's).
    private static final double SIGMA = 0.9;
    private static final double INITIAL_FRACTION = 0.02;
    private static final int RADIUS = 3; // ceil(3 * SIGMA)

    private volatile int[] words;

    private RtBlueNoise() {
    }

    /**
     * The packed tile, generated on first use. Each word is
     * {@code R | G << 8 | B << 16 | A << 24}, with every byte an 8-bit blue-noise rank.
     */
    public int[] words() {
        int[] local = words;
        if (local == null) {
            local = generate();
            words = local;
        }
        return local;
    }

    private static int[] generate() {
        int[] red = rankMask(0x9E3779B97F4A7C15L);
        int[] green = rankMask(0xC2B2AE3D27D4EB4FL);
        int[] blue = rankMask(0x165667B19E3779F9L);
        int[] alpha = rankMask(0xD6E8FEB86659FD93L);

        int[] result = new int[TEXELS];
        for (int i = 0; i < TEXELS; i++) {
            result[i] = red[i]
                    | (green[i] << 8)
                    | (blue[i] << 16)
                    | (alpha[i] << 24);
        }
        return result;
    }

    /**
     * Ulichney's void-and-cluster ranking on a toroidal grid.
     *
     * <p>The initial random pattern is the prototype BOTH ranking passes start from — that is what
     * makes every thresholded subset of the result a blue-noise point set. Phase 1 removes the
     * tightest clusters one at a time, assigning ranks from the initial count-1 down to zero; phase 2
     * fills the largest voids, assigning the remaining ranks upward. The result is a permutation of
     * {@code 0..TEXELS-1}.
     */
    private static int[] rankMask(long seed) {
        Random random = new Random(seed);
        int[] order = new int[TEXELS];
        for (int i = 0; i < TEXELS; i++) {
            order[i] = i;
        }
        for (int i = TEXELS - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int tmp = order[i];
            order[i] = order[j];
            order[j] = tmp;
        }
        int initialCount = Math.max(1, (int) (TEXELS * INITIAL_FRACTION));

        boolean[] prototype = new boolean[TEXELS];
        for (int i = 0; i < initialCount; i++) {
            prototype[order[i]] = true;
        }

        // Periodic Gaussian kernel, precomputed once and shared by every energy update.
        int side = RADIUS * 2 + 1;
        int[] kernelDx = new int[side * side];
        int[] kernelDy = new int[side * side];
        double[] kernelWeight = new double[side * side];
        int kernelIndex = 0;
        for (int dy = -RADIUS; dy <= RADIUS; dy++) {
            for (int dx = -RADIUS; dx <= RADIUS; dx++) {
                kernelDx[kernelIndex] = dx;
                kernelDy[kernelIndex] = dy;
                kernelWeight[kernelIndex] = Math.exp(-(dx * dx + dy * dy) / (2.0 * SIGMA * SIGMA));
                kernelIndex++;
            }
        }

        double[] prototypeEnergy = new double[TEXELS];
        for (int i = 0; i < TEXELS; i++) {
            if (prototype[i]) {
                addKernel(prototypeEnergy, i % TILE, i / TILE, kernelDx, kernelDy, kernelWeight, 1.0);
            }
        }

        int[] ranks = new int[TEXELS];

        // Phase 1: strip clusters from the prototype, highest rank first.
        boolean[] occupied = prototype.clone();
        double[] energy = prototypeEnergy.clone();
        for (int rank = initialCount - 1; rank >= 0; rank--) {
            int cluster = -1;
            double clusterEnergy = Double.NEGATIVE_INFINITY;
            for (int i = 0; i < TEXELS; i++) {
                if (occupied[i] && energy[i] > clusterEnergy) {
                    clusterEnergy = energy[i];
                    cluster = i;
                }
            }
            occupied[cluster] = false;
            addKernel(energy, cluster % TILE, cluster / TILE, kernelDx, kernelDy, kernelWeight, -1.0);
            ranks[cluster] = rank;
        }

        // Phase 2: fill voids from the same prototype, lowest rank first.
        occupied = prototype.clone();
        energy = prototypeEnergy.clone();
        for (int rank = initialCount; rank < TEXELS; rank++) {
            int voidCell = -1;
            double voidEnergy = Double.POSITIVE_INFINITY;
            for (int i = 0; i < TEXELS; i++) {
                if (!occupied[i] && energy[i] < voidEnergy) {
                    voidEnergy = energy[i];
                    voidCell = i;
                }
            }
            occupied[voidCell] = true;
            addKernel(energy, voidCell % TILE, voidCell / TILE, kernelDx, kernelDy, kernelWeight, 1.0);
            ranks[voidCell] = rank;
        }

        // 4096 ranks -> 256 byte levels. The top four bits of the rank ARE the level: 4096 / 256 = 16
        // exactly, so every level appears exactly 16 times and the tile's byte histogram is flat.
        // Keeping the low four rank bits would only add sub-level detail no 8-bit consumer can see.
        int[] bytes = new int[TEXELS];
        for (int i = 0; i < TEXELS; i++) {
            bytes[i] = ranks[i] >> 4;
        }
        return bytes;
    }

    private static void addKernel(double[] energy, int x, int y,
                                  int[] kernelDx, int[] kernelDy, double[] kernelWeight, double sign) {
        for (int k = 0; k < kernelWeight.length; k++) {
            int px = (x + kernelDx[k] + TILE) & (TILE - 1); // TILE is a power of two
            int py = (y + kernelDy[k] + TILE) & (TILE - 1);
            energy[py * TILE + px] += sign * kernelWeight[k];
        }
    }
}
