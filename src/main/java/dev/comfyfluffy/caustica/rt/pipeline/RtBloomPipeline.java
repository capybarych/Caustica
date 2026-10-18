package dev.comfyfluffy.caustica.rt.pipeline;

import dev.comfyfluffy.caustica.CausticaConfig;
import dev.comfyfluffy.caustica.rt.RtContext;
import dev.comfyfluffy.caustica.rt.RtDebugLabels;
import dev.comfyfluffy.caustica.rt.accel.RtImage;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;
import org.lwjgl.vulkan.VK10;
import org.lwjgl.vulkan.VkCommandBuffer;
import org.lwjgl.vulkan.VkComputePipelineCreateInfo;
import org.lwjgl.vulkan.VkDescriptorImageInfo;
import org.lwjgl.vulkan.VkDescriptorPoolCreateInfo;
import org.lwjgl.vulkan.VkDescriptorPoolSize;
import org.lwjgl.vulkan.VkDescriptorSetAllocateInfo;
import org.lwjgl.vulkan.VkDescriptorSetLayoutBinding;
import org.lwjgl.vulkan.VkDescriptorSetLayoutCreateInfo;
import org.lwjgl.vulkan.VkDevice;
import org.lwjgl.vulkan.VkMemoryBarrier;
import org.lwjgl.vulkan.VkPipelineLayoutCreateInfo;
import org.lwjgl.vulkan.VkPipelineShaderStageCreateInfo;
import org.lwjgl.vulkan.VkPushConstantRange;
import org.lwjgl.vulkan.VkSamplerCreateInfo;
import org.lwjgl.vulkan.VkShaderModuleCreateInfo;
import org.lwjgl.vulkan.VkWriteDescriptorSet;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.LongBuffer;
import java.util.Arrays;

import static dev.comfyfluffy.caustica.rt.RtContext.check;

/**
 * Multi-scale screen-space HDR Bloom compute pipeline.
 *
 * <p>Operates in linear HDR space at display resolution between {@link RtExposure} and
 * {@link RtDisplayPipeline}:
 * <ol>
 *   <li><b>Prefilter + Downsample Pyramid</b> ({@code bloom_downsample.comp}): normalizes scene
 *       radiance by the current exposure scale, extracts highlights above the configured threshold
 *       and quadratic soft knee (with normalized Karis anti-firefly weighting on High/Ultra), and
 *       progressively downsamples through up to {@value #MAX_MIPS} half-resolution mip levels
 *       ({@code 1/2} down to {@code 1/64}).</li>
 *   <li><b>Progressive Tent Upsample</b> ({@code bloom_upsample.comp}): upsamples and accumulates
 *       from the coarsest active mip back up to mip 0 using an energy-conserving convex combination
 *       controlled by the configured radius/spread and quality preset.</li>
 *   <li><b>Scene-Linear HDR Composite</b> ({@code bloom_composite.comp}): upsamples mip 0 to full
 *       display resolution, applies luminance-preserving color influence and intensity scaling, and
 *       adds the bloom contribution back into the linear HDR buffer before SDR/HDR tone mapping.</li>
 * </ol>
 */
public final class RtBloomPipeline {
    private static final String SHADER_DIR = "/caustica/rt/";

    /** Maximum number of bloom mip levels (Low=3, Medium=4, High=5, Ultra=6). */
    public static final int MAX_MIPS = 6;

    /** Push constants for {@code bloom_downsample.comp}: 6 ints + 2 floats = 32 bytes. */
    private static final int DOWNSAMPLE_PUSH_BYTES = 8 * Integer.BYTES;
    /** Push constants for {@code bloom_upsample.comp}: 5 ints + 1 float = 24 bytes. */
    private static final int UPSAMPLE_PUSH_BYTES = 6 * Integer.BYTES;
    /** Push constants for {@code bloom_composite.comp}: 5 ints + 3 floats = 32 bytes. */
    private static final int COMPOSITE_PUSH_BYTES = 8 * Integer.BYTES;

    private final RtContext ctx;
    private final long sampler;
    private final long descriptorPool;

    private final long downsampleDsl;
    private final long downsampleLayout;
    private final long downsamplePipeline;
    private final long[] downsampleSets = new long[MAX_MIPS];
    private final long[][] boundDownsampleViews = new long[MAX_MIPS][3];

    private final long upsampleDsl;
    private final long upsampleLayout;
    private final long upsamplePipeline;
    private final long[] upsampleSets = new long[MAX_MIPS - 1];
    private final long[][] boundUpsampleViews = new long[MAX_MIPS - 1][2];

    private final long compositeDsl;
    private final long compositeLayout;
    private final long compositePipeline;
    private final long compositeSet;
    private final long[] boundCompositeViews = new long[3];

    private final RtImage[] mips = new RtImage[MAX_MIPS];
    private int allocatedWidth = -1;
    private int allocatedHeight = -1;
    private boolean destroyed;

    private RtBloomPipeline(
            RtContext ctx,
            long sampler,
            long descriptorPool,
            long downsampleDsl,
            long downsampleLayout,
            long downsamplePipeline,
            long[] downsampleSets,
            long upsampleDsl,
            long upsampleLayout,
            long upsamplePipeline,
            long[] upsampleSets,
            long compositeDsl,
            long compositeLayout,
            long compositePipeline,
            long compositeSet
    ) {
        this.ctx = ctx;
        this.sampler = sampler;
        this.descriptorPool = descriptorPool;
        this.downsampleDsl = downsampleDsl;
        this.downsampleLayout = downsampleLayout;
        this.downsamplePipeline = downsamplePipeline;
        System.arraycopy(downsampleSets, 0, this.downsampleSets, 0, MAX_MIPS);
        this.upsampleDsl = upsampleDsl;
        this.upsampleLayout = upsampleLayout;
        this.upsamplePipeline = upsamplePipeline;
        System.arraycopy(upsampleSets, 0, this.upsampleSets, 0, MAX_MIPS - 1);
        this.compositeDsl = compositeDsl;
        this.compositeLayout = compositeLayout;
        this.compositePipeline = compositePipeline;
        this.compositeSet = compositeSet;
    }

    public static RtBloomPipeline create(RtContext ctx) {
        VkDevice vk = ctx.vk();
        try (MemoryStack stack = MemoryStack.stackPush()) {
            LongBuffer p = stack.mallocLong(1);

            VkSamplerCreateInfo sci = VkSamplerCreateInfo.calloc(stack).sType$Default()
                    .magFilter(VK10.VK_FILTER_LINEAR)
                    .minFilter(VK10.VK_FILTER_LINEAR)
                    .mipmapMode(VK10.VK_SAMPLER_MIPMAP_MODE_NEAREST)
                    .addressModeU(VK10.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .addressModeV(VK10.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .addressModeW(VK10.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .minLod(0.0f)
                    .maxLod(0.0f);
            check(VK10.vkCreateSampler(vk, sci, null, p), "vkCreateSampler(bloom linear)");
            long sampler = p.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_SAMPLER, sampler, "bloom linear sampler");

            // Downsample layout: binding 0 = sampler2D srcTex, binding 1 = exposureImage, binding 2 = dstImage
            long downsampleDsl = createSetLayout(ctx, stack, "bloom downsample",
                    VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);

            // Upsample layout: binding 0 = sampler2D coarseTex, binding 1 = fineImage
            long upsampleDsl = createSetLayout(ctx, stack, "bloom upsample",
                    VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);

            // Composite layout: binding 0 = sampler2D bloomTex, binding 1 = exposureImage, binding 2 = sceneImage
            long compositeDsl = createSetLayout(ctx, stack, "bloom composite",
                    VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);

            int totalSets = MAX_MIPS + (MAX_MIPS - 1) + 1;
            int totalSampled = MAX_MIPS + (MAX_MIPS - 1) + 1;
            int totalStorage = (MAX_MIPS * 2) + (MAX_MIPS - 1) + 2;

            VkDescriptorPoolSize.Buffer poolSizes = VkDescriptorPoolSize.calloc(2, stack);
            poolSizes.get(0).type(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).descriptorCount(totalSampled);
            poolSizes.get(1).type(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).descriptorCount(totalStorage);
            VkDescriptorPoolCreateInfo dpci = VkDescriptorPoolCreateInfo.calloc(stack).sType$Default()
                    .maxSets(totalSets)
                    .pPoolSizes(poolSizes);
            check(VK10.vkCreateDescriptorPool(vk, dpci, null, p), "vkCreateDescriptorPool(bloom)");
            long pool = p.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_POOL, pool, "bloom descriptor pool");

            long[] downsampleSets = allocateSets(ctx, stack, pool, downsampleDsl, MAX_MIPS, "bloom downsample");
            long[] upsampleSets = allocateSets(ctx, stack, pool, upsampleDsl, MAX_MIPS - 1, "bloom upsample");
            long compositeSet = allocateSets(ctx, stack, pool, compositeDsl, 1, "bloom composite")[0];

            long downsampleLayout = createPipelineLayout(ctx, stack, downsampleDsl, DOWNSAMPLE_PUSH_BYTES, "bloom downsample");
            long upsampleLayout = createPipelineLayout(ctx, stack, upsampleDsl, UPSAMPLE_PUSH_BYTES, "bloom upsample");
            long compositeLayout = createPipelineLayout(ctx, stack, compositeDsl, COMPOSITE_PUSH_BYTES, "bloom composite");

            long downsamplePipeline = createComputePipeline(
                    ctx, stack, downsampleLayout, "bloom_downsample.comp.spv", "bloom downsample");
            long upsamplePipeline = createComputePipeline(
                    ctx, stack, upsampleLayout, "bloom_upsample.comp.spv", "bloom upsample");
            long compositePipeline = createComputePipeline(
                    ctx, stack, compositeLayout, "bloom_composite.comp.spv", "bloom composite");

            return new RtBloomPipeline(
                    ctx, sampler, pool,
                    downsampleDsl, downsampleLayout, downsamplePipeline, downsampleSets,
                    upsampleDsl, upsampleLayout, upsamplePipeline, upsampleSets,
                    compositeDsl, compositeLayout, compositePipeline, compositeSet
            );
        }
    }

    /**
     * Ensure the internal bloom mip pyramid is sized for {@code displayWidth x displayHeight}.
     * Caller must ensure the device is idle if previous mip images are being replaced.
     */
    public void ensureSize(int displayWidth, int displayHeight) {
        if (displayWidth <= 0 || displayHeight <= 0) {
            return;
        }
        if (allocatedWidth == displayWidth && allocatedHeight == displayHeight && mips[0] != null) {
            return;
        }
        destroyMips();
        int w = displayWidth;
        int h = displayHeight;
        for (int i = 0; i < MAX_MIPS; i++) {
            w = Math.max(1, (w + 1) >> 1);
            h = Math.max(1, (h + 1) >> 1);
            mips[i] = ctx.createStorageImage(
                    w, h, VK10.VK_FORMAT_R16G16B16A16_SFLOAT, "bloom mip " + i + " " + w + "x" + h);
        }
        allocatedWidth = displayWidth;
        allocatedHeight = displayHeight;
        invalidateBindings();
    }

    public boolean readyForSize(int displayWidth, int displayHeight) {
        return !destroyed && mips[0] != null && allocatedWidth == displayWidth && allocatedHeight == displayHeight;
    }

    /**
     * Drop cached descriptor view handles so the next dispatch rewrites all descriptor sets.
     */
    public void invalidateBindings() {
        for (long[] row : boundDownsampleViews) {
            Arrays.fill(row, 0L);
        }
        for (long[] row : boundUpsampleViews) {
            Arrays.fill(row, 0L);
        }
        Arrays.fill(boundCompositeViews, 0L);
    }

    /**
     * Record the multi-scale bloom passes into {@code cmd} when Bloom is enabled.
     * When Bloom is disabled or intensity is zero, this method returns immediately without
     * recording any GPU work or altering {@code sceneHdr}.
     */
    public void record(VkCommandBuffer cmd, MemoryStack stack, RtImage sceneHdr, RtImage exposureImage) {
        if (destroyed || sceneHdr == null || exposureImage == null) {
            return;
        }
        if (!CausticaConfig.Rt.Bloom.ENABLED.value()) {
            return;
        }
        float intensity = CausticaConfig.Rt.Bloom.INTENSITY.value();
        if (intensity <= 0.0f) {
            return;
        }
        if (!readyForSize(sceneHdr.width, sceneHdr.height)) {
            return;
        }

        int quality = CausticaConfig.Rt.Bloom.qualityIndex();
        int targetMips = CausticaConfig.Rt.Bloom.mipCountForQuality(quality);
        int activeMips = computeActiveMips(targetMips);
        float threshold = CausticaConfig.Rt.Bloom.THRESHOLD.value();
        float softKnee = CausticaConfig.Rt.Bloom.SOFT_KNEE.value();
        float radius = CausticaConfig.Rt.Bloom.RADIUS.value();
        float colorInfluence = CausticaConfig.Rt.Bloom.COLOR_INFLUENCE.value();

        try (RtDebugLabels.Scope ignored = RtDebugLabels.scope(ctx, cmd, "rt bloom")) {
            // 1. Progressive Downsample Pyramid (sceneHdr -> mip[0] -> mip[1] -> ... -> mip[activeMips - 1])
            VK10.vkCmdBindPipeline(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, downsamplePipeline);
            ByteBuffer downPush = stack.malloc(DOWNSAMPLE_PUSH_BYTES);
            for (int level = 0; level < activeMips; level++) {
                RtImage src = (level == 0) ? sceneHdr : mips[level - 1];
                RtImage dst = mips[level];
                bindDownsampleSet(stack, level, src.view, exposureImage.view, dst.view);
                VK10.vkCmdBindDescriptorSets(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, downsampleLayout, 0,
                        stack.longs(downsampleSets[level]), null);

                downPush.putInt(0, src.width);
                downPush.putInt(4, src.height);
                downPush.putInt(8, dst.width);
                downPush.putInt(12, dst.height);
                downPush.putInt(16, level);
                downPush.putInt(20, quality);
                downPush.putFloat(24, threshold);
                downPush.putFloat(28, softKnee);
                VK10.vkCmdPushConstants(cmd, downsampleLayout, VK10.VK_SHADER_STAGE_COMPUTE_BIT, 0, downPush);
                VK10.vkCmdDispatch(cmd, (dst.width + 15) / 16, (dst.height + 15) / 16, 1);

                barrierComputeReadWrite(cmd, stack);
            }

            // 2. Progressive Tent Upsample (mip[activeMips - 1] -> ... -> mip[0])
            if (activeMips > 1) {
                VK10.vkCmdBindPipeline(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, upsamplePipeline);
                ByteBuffer upPush = stack.malloc(UPSAMPLE_PUSH_BYTES);
                for (int level = activeMips - 2; level >= 0; level--) {
                    RtImage coarse = mips[level + 1];
                    RtImage fine = mips[level];
                    bindUpsampleSet(stack, level, coarse.view, fine.view);
                    VK10.vkCmdBindDescriptorSets(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, upsampleLayout, 0,
                            stack.longs(upsampleSets[level]), null);

                    upPush.putInt(0, coarse.width);
                    upPush.putInt(4, coarse.height);
                    upPush.putInt(8, fine.width);
                    upPush.putInt(12, fine.height);
                    upPush.putInt(16, quality);
                    upPush.putFloat(20, radius);
                    VK10.vkCmdPushConstants(cmd, upsampleLayout, VK10.VK_SHADER_STAGE_COMPUTE_BIT, 0, upPush);
                    VK10.vkCmdDispatch(cmd, (fine.width + 15) / 16, (fine.height + 15) / 16, 1);

                    barrierComputeReadWrite(cmd, stack);
                }
            }

            // 3. Final Composite into sceneHdr (mip[0] + exposureImage + sceneHdr -> sceneHdr)
            RtImage bloomMip0 = mips[0];
            bindCompositeSet(stack, bloomMip0.view, exposureImage.view, sceneHdr.view);
            VK10.vkCmdBindPipeline(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, compositePipeline);
            VK10.vkCmdBindDescriptorSets(cmd, VK10.VK_PIPELINE_BIND_POINT_COMPUTE, compositeLayout, 0,
                    stack.longs(compositeSet), null);

            ByteBuffer compPush = stack.malloc(COMPOSITE_PUSH_BYTES);
            compPush.putInt(0, bloomMip0.width);
            compPush.putInt(4, bloomMip0.height);
            compPush.putInt(8, sceneHdr.width);
            compPush.putInt(12, sceneHdr.height);
            compPush.putInt(16, quality);
            compPush.putFloat(20, intensity);
            compPush.putFloat(24, radius);
            compPush.putFloat(28, colorInfluence);
            VK10.vkCmdPushConstants(cmd, compositeLayout, VK10.VK_SHADER_STAGE_COMPUTE_BIT, 0, compPush);
            VK10.vkCmdDispatch(cmd, (sceneHdr.width + 15) / 16, (sceneHdr.height + 15) / 16, 1);

            barrierComputeReadWrite(cmd, stack);
        }
    }

    private int computeActiveMips(int targetMips) {
        int clampedTarget = Math.clamp(targetMips, 1, MAX_MIPS);
        int count = 1;
        while (count < clampedTarget && mips[count] != null
                && (mips[count - 1].width > 1 || mips[count - 1].height > 1)) {
            count++;
        }
        return count;
    }

    private static void barrierComputeReadWrite(VkCommandBuffer cmd, MemoryStack stack) {
        VkMemoryBarrier.Buffer barrier = VkMemoryBarrier.calloc(1, stack);
        barrier.get(0).sType$Default()
                .srcAccessMask(VK10.VK_ACCESS_SHADER_WRITE_BIT)
                .dstAccessMask(VK10.VK_ACCESS_SHADER_READ_BIT | VK10.VK_ACCESS_SHADER_WRITE_BIT);
        VK10.vkCmdPipelineBarrier(cmd,
                VK10.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK10.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, barrier, null, null);
    }

    private void bindDownsampleSet(MemoryStack stack, int level, long srcView, long exposureView, long dstView) {
        long[] bound = boundDownsampleViews[level];
        if (bound[0] == srcView && bound[1] == exposureView && bound[2] == dstView) {
            return;
        }
        VkDescriptorImageInfo.Buffer srcInfo = VkDescriptorImageInfo.calloc(1, stack);
        srcInfo.get(0).sampler(sampler).imageView(srcView).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);

        VkDescriptorImageInfo.Buffer expInfo = VkDescriptorImageInfo.calloc(1, stack);
        expInfo.get(0).imageView(exposureView).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);

        VkDescriptorImageInfo.Buffer dstInfo = VkDescriptorImageInfo.calloc(1, stack);
        dstInfo.get(0).imageView(dstView).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);

        long set = downsampleSets[level];
        VkWriteDescriptorSet.Buffer writes = VkWriteDescriptorSet.calloc(3, stack);
        writes.get(0).sType$Default().dstSet(set).dstBinding(0).descriptorCount(1)
                .descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).pImageInfo(srcInfo);
        writes.get(1).sType$Default().dstSet(set).dstBinding(1).descriptorCount(1)
                .descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).pImageInfo(expInfo);
        writes.get(2).sType$Default().dstSet(set).dstBinding(2).descriptorCount(1)
                .descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).pImageInfo(dstInfo);
        VK10.vkUpdateDescriptorSets(ctx.vk(), writes, null);

        bound[0] = srcView;
        bound[1] = exposureView;
        bound[2] = dstView;
    }

    private void bindUpsampleSet(MemoryStack stack, int level, long coarseView, long fineView) {
        long[] bound = boundUpsampleViews[level];
        if (bound[0] == coarseView && bound[1] == fineView) {
            return;
        }
        VkDescriptorImageInfo.Buffer coarseInfo = VkDescriptorImageInfo.calloc(1, stack);
        coarseInfo.get(0).sampler(sampler).imageView(coarseView).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);

        VkDescriptorImageInfo.Buffer fineInfo = VkDescriptorImageInfo.calloc(1, stack);
        fineInfo.get(0).imageView(fineView).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);

        long set = upsampleSets[level];
        VkWriteDescriptorSet.Buffer writes = VkWriteDescriptorSet.calloc(2, stack);
        writes.get(0).sType$Default().dstSet(set).dstBinding(0).descriptorCount(1)
                .descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).pImageInfo(coarseInfo);
        writes.get(1).sType$Default().dstSet(set).dstBinding(1).descriptorCount(1)
                .descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).pImageInfo(fineInfo);
        VK10.vkUpdateDescriptorSets(ctx.vk(), writes, null);

        bound[0] = coarseView;
        bound[1] = fineView;
    }

    private void bindCompositeSet(MemoryStack stack, long bloomView, long exposureView, long sceneView) {
        if (boundCompositeViews[0] == bloomView
                && boundCompositeViews[1] == exposureView
                && boundCompositeViews[2] == sceneView) {
            return;
        }
        VkDescriptorImageInfo.Buffer bloomInfo = VkDescriptorImageInfo.calloc(1, stack);
        bloomInfo.get(0).sampler(sampler).imageView(bloomView).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);

        VkDescriptorImageInfo.Buffer expInfo = VkDescriptorImageInfo.calloc(1, stack);
        expInfo.get(0).imageView(exposureView).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);

        VkDescriptorImageInfo.Buffer sceneInfo = VkDescriptorImageInfo.calloc(1, stack);
        sceneInfo.get(0).imageView(sceneView).imageLayout(VK10.VK_IMAGE_LAYOUT_GENERAL);

        VkWriteDescriptorSet.Buffer writes = VkWriteDescriptorSet.calloc(3, stack);
        writes.get(0).sType$Default().dstSet(compositeSet).dstBinding(0).descriptorCount(1)
                .descriptorType(VK10.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).pImageInfo(bloomInfo);
        writes.get(1).sType$Default().dstSet(compositeSet).dstBinding(1).descriptorCount(1)
                .descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).pImageInfo(expInfo);
        writes.get(2).sType$Default().dstSet(compositeSet).dstBinding(2).descriptorCount(1)
                .descriptorType(VK10.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).pImageInfo(sceneInfo);
        VK10.vkUpdateDescriptorSets(ctx.vk(), writes, null);

        boundCompositeViews[0] = bloomView;
        boundCompositeViews[1] = exposureView;
        boundCompositeViews[2] = sceneView;
    }

    private void destroyMips() {
        for (int i = 0; i < MAX_MIPS; i++) {
            if (mips[i] != null) {
                mips[i].destroy();
                mips[i] = null;
            }
        }
        allocatedWidth = -1;
        allocatedHeight = -1;
    }

    public void destroy() {
        if (destroyed) {
            return;
        }
        destroyMips();
        VkDevice vk = ctx.vk();
        VK10.vkDestroyPipeline(vk, downsamplePipeline, null);
        VK10.vkDestroyPipeline(vk, upsamplePipeline, null);
        VK10.vkDestroyPipeline(vk, compositePipeline, null);
        VK10.vkDestroyPipelineLayout(vk, downsampleLayout, null);
        VK10.vkDestroyPipelineLayout(vk, upsampleLayout, null);
        VK10.vkDestroyPipelineLayout(vk, compositeLayout, null);
        VK10.vkDestroyDescriptorPool(vk, descriptorPool, null);
        VK10.vkDestroyDescriptorSetLayout(vk, downsampleDsl, null);
        VK10.vkDestroyDescriptorSetLayout(vk, upsampleDsl, null);
        VK10.vkDestroyDescriptorSetLayout(vk, compositeDsl, null);
        VK10.vkDestroySampler(vk, sampler, null);
        destroyed = true;
    }

    private static long createSetLayout(RtContext ctx, MemoryStack stack, String label, int... types) {
        VkDescriptorSetLayoutBinding.Buffer binds = VkDescriptorSetLayoutBinding.calloc(types.length, stack);
        for (int i = 0; i < types.length; i++) {
            binds.get(i).binding(i).descriptorType(types[i])
                    .descriptorCount(1).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT);
        }
        VkDescriptorSetLayoutCreateInfo dslci = VkDescriptorSetLayoutCreateInfo.calloc(stack)
                .sType$Default().pBindings(binds);
        LongBuffer p = stack.mallocLong(1);
        check(VK10.vkCreateDescriptorSetLayout(ctx.vk(), dslci, null, p),
                "vkCreateDescriptorSetLayout(" + label + ")");
        long dsl = p.get(0);
        RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT, dsl, label + " descriptor set layout");
        return dsl;
    }

    private static long[] allocateSets(RtContext ctx, MemoryStack stack, long pool, long dsl, int count, String label) {
        LongBuffer layouts = stack.mallocLong(count);
        for (int i = 0; i < count; i++) {
            layouts.put(i, dsl);
        }
        VkDescriptorSetAllocateInfo dsai = VkDescriptorSetAllocateInfo.calloc(stack).sType$Default()
                .descriptorPool(pool).pSetLayouts(layouts);
        LongBuffer pSets = stack.mallocLong(count);
        check(VK10.vkAllocateDescriptorSets(ctx.vk(), dsai, pSets), "vkAllocateDescriptorSets(" + label + ")");
        long[] sets = new long[count];
        for (int i = 0; i < count; i++) {
            sets[i] = pSets.get(i);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_DESCRIPTOR_SET, sets[i], label + " descriptor set " + i);
        }
        return sets;
    }

    private static long createPipelineLayout(RtContext ctx, MemoryStack stack, long dsl, int pushBytes, String label) {
        VkPushConstantRange.Buffer pushRange = VkPushConstantRange.calloc(1, stack);
        pushRange.get(0).stageFlags(VK10.VK_SHADER_STAGE_COMPUTE_BIT).offset(0).size(pushBytes);
        VkPipelineLayoutCreateInfo plci = VkPipelineLayoutCreateInfo.calloc(stack).sType$Default()
                .pSetLayouts(stack.longs(dsl)).pPushConstantRanges(pushRange);
        LongBuffer p = stack.mallocLong(1);
        check(VK10.vkCreatePipelineLayout(ctx.vk(), plci, null, p), "vkCreatePipelineLayout(" + label + ")");
        long layout = p.get(0);
        RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_PIPELINE_LAYOUT, layout, label + " pipeline layout");
        return layout;
    }

    private static long createComputePipeline(
            RtContext ctx, MemoryStack stack, long layout, String spvName, String label
    ) {
        VkDevice vk = ctx.vk();
        long module = loadModule(vk, stack, spvName);
        RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_SHADER_MODULE, module, label + " shader module");
        try {
            VkPipelineShaderStageCreateInfo stage = VkPipelineShaderStageCreateInfo.calloc(stack).sType$Default()
                    .stage(VK10.VK_SHADER_STAGE_COMPUTE_BIT).module(module).pName(stack.UTF8("main"));
            VkComputePipelineCreateInfo.Buffer cpci = VkComputePipelineCreateInfo.calloc(1, stack);
            cpci.get(0).sType$Default().stage(stage).layout(layout);
            LongBuffer pPipeline = stack.mallocLong(1);
            check(VK10.vkCreateComputePipelines(vk, VK10.VK_NULL_HANDLE, cpci, null, pPipeline),
                    "vkCreateComputePipelines(" + label + ")");
            long pipeline = pPipeline.get(0);
            RtDebugLabels.name(ctx, VK10.VK_OBJECT_TYPE_PIPELINE, pipeline, label + " compute pipeline");
            return pipeline;
        } finally {
            VK10.vkDestroyShaderModule(vk, module, null);
        }
    }

    private static long loadModule(VkDevice vk, MemoryStack stack, String name) {
        byte[] bytes;
        try (InputStream in = RtBloomPipeline.class.getResourceAsStream(SHADER_DIR + name)) {
            if (in == null) {
                throw new IllegalStateException("missing SPIR-V resource: " + SHADER_DIR + name);
            }
            bytes = in.readAllBytes();
        } catch (IOException e) {
            throw new IllegalStateException("failed to read SPIR-V resource: " + SHADER_DIR + name, e);
        }
        ByteBuffer code = MemoryUtil.memAlloc(bytes.length).put(bytes);
        code.flip();
        try {
            VkShaderModuleCreateInfo smci = VkShaderModuleCreateInfo.calloc(stack).sType$Default().pCode(code);
            LongBuffer pModule = stack.mallocLong(1);
            check(VK10.vkCreateShaderModule(vk, smci, null, pModule), "vkCreateShaderModule(" + name + ")");
            return pModule.get(0);
        } finally {
            MemoryUtil.memFree(code);
        }
    }
}
