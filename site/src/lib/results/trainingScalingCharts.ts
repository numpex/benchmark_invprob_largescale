// Vega-Lite specs for the distributed-training finding.
//
// Palette notes: the categorical slots and the ordinal blue ramp below were
// checked with the data-viz validator against this page's white chart surface
// (categorical 3-slot all-pairs and 5-slot adjacent both pass; the ordinal ramp
// is monotone in lightness with a light end clearing the surface). Several
// slots sit under 3:1 contrast, so every figure on the page ships its
// "View figure data" table as the required relief.

export const config = {
  background: null,
  font: "Inter, ui-sans-serif, system-ui, sans-serif",
  view: { stroke: null },
  title: {
    fontSize: 17,
    subtitleFontSize: 13,
    subtitleColor: "#50617d",
  },
  axis: {
    domainColor: "#8290a8",
    gridColor: "#e5e8e2",
    labelColor: "#50617d",
    titleColor: "#20304e",
    titleFontWeight: 600,
    labelFontSize: 13,
    titleFontSize: 14,
  },
  legend: {
    labelColor: "#50617d",
    titleColor: "#20304e",
    orient: "bottom",
    direction: "horizontal",
    labelFontSize: 13,
    titleFontSize: 14,
    symbolSize: 110,
    symbolOpacity: 1,
  },
  header: {
    labelFontSize: 15,
    titleFontSize: 15,
  },
};

// Categorical slots, assigned in fixed order and never cycled.
export const series = ["#2a78d6", "#eb6834", "#1baf7a"];

// Ordinal ramp for per-GPU workload, which is an ordered magnitude rather than
// an identity — one hue, light (small workload) to dark (large workload).
const workloadRamp = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"];

export const guideColor = "#8290a8";
export const annotationColor = "#50617d";

/** Per-GPU memory of the cluster's V100 cards, in MiB. */
export const DEVICE_MEMORY_MB = 32768;

const gpuAxis = {
  field: "gpuCount",
  type: "quantitative",
  title: "GPUs",
  scale: { type: "log", base: 2 },
  axis: { values: [1, 2, 4, 8, 16, 32, 64], format: "d" },
};

const imageSizeColor = {
  field: "imageSize",
  type: "nominal",
  title: "Image size",
  scale: {
    domain: ["1024x1024", "4096x4096", "8192x8192"],
    range: series,
  },
};

const workloadColor = {
  field: "localWorkloadMpix",
  type: "ordinal",
  title: "Mpix / GPU",
  scale: { range: workloadRamp },
  legend: { format: ".2f" },
};

const checkpointColor = {
  field: "checkpointMode",
  type: "nominal",
  title: "Checkpointing",
  scale: { domain: ["always", "never"], range: [series[0], series[1]] },
};

export const lineMark = {
  type: "line" as const,
  point: { filled: true, size: 85 },
  strokeWidth: 2,
};

/** Dashed horizontal reference at the ideal value of a ratio axis. */
export const idealRule = (value: number, label: string) => [
  {
    mark: { type: "rule", strokeDash: [6, 5], color: guideColor },
    encoding: { y: { datum: value } },
  },
  {
    mark: {
      type: "text",
      align: "left",
      baseline: "bottom",
      dx: 6,
      dy: -4,
      color: annotationColor,
      fontSize: 12,
    },
    encoding: {
      y: { datum: value },
      x: { datum: 1 },
      text: { datum: label },
    },
  },
];

export const strongScalingSpec = {
  $schema: "https://vega.github.io/schema/vega-lite/v6.json",
  description:
    "Training parallel efficiency against GPU count, one line per image size.",
  title: {
    text: "Strong scaling: efficiency holds while each GPU still has work",
    subtitle:
      "Parallel efficiency of the per-step forward + backward time, normalized to the smallest-GPU run of each image size.",
  },
  width: 620,
  height: 400,
  config,
  layer: [
    ...idealRule(100, "perfect scaling"),
    {
      mark: lineMark,
      encoding: {
        x: gpuAxis,
        y: {
          field: "efficiencyPct",
          type: "quantitative",
          title: "Parallel efficiency (%)",
          scale: { domain: [0, 110] },
        },
        color: imageSizeColor,
        tooltip: [
          { field: "imageSize", type: "nominal", title: "Image size" },
          { field: "gpuCount", type: "quantitative", title: "GPUs" },
          {
            field: "timeSec",
            type: "quantitative",
            title: "Step time (s)",
            format: ".2f",
          },
          {
            field: "speedup",
            type: "quantitative",
            title: "Speedup",
            format: ".2f",
          },
          {
            field: "efficiencyPct",
            type: "quantitative",
            title: "Efficiency (%)",
            format: ".1f",
          },
        ],
      },
    },
    {
      // Direct labels, so identity never rests on color alone.
      transform: [
        {
          window: [{ op: "rank", as: "gpuRank" }],
          groupby: ["imageSize"],
          sort: [{ field: "gpuCount", order: "descending" }],
        },
        { filter: "datum.gpuRank === 1" },
      ],
      mark: {
        type: "text",
        align: "right",
        baseline: "top",
        dy: 12,
        fontSize: 12,
        fontWeight: 600,
      },
      encoding: {
        x: gpuAxis,
        y: { field: "efficiencyPct", type: "quantitative" },
        text: { field: "imageSize", type: "nominal" },
        color: imageSizeColor,
      },
    },
  ],
};

export const weakScalingSpec = {
  $schema: "https://vega.github.io/schema/vega-lite/v6.json",
  description:
    "Weak-scaling efficiency against GPU count, one line per per-GPU workload.",
  title: {
    text: "Weak scaling: the thinner the per-GPU workload, the worse it holds",
    subtitle:
      "Each line keeps megapixels-per-GPU fixed while growing both the image and the machine.",
  },
  width: 620,
  height: 400,
  config,
  layer: [
    ...idealRule(100, "perfect scaling"),
    {
      mark: lineMark,
      encoding: {
        x: gpuAxis,
        y: {
          field: "efficiencyPct",
          type: "quantitative",
          title: "Weak-scaling efficiency (%)",
          scale: { domain: [0, 110] },
        },
        color: workloadColor,
        tooltip: [
          {
            field: "localWorkloadMpix",
            type: "quantitative",
            title: "Mpix / GPU",
            format: ".2f",
          },
          { field: "imageSize", type: "nominal", title: "Image size" },
          { field: "gpuCount", type: "quantitative", title: "GPUs" },
          {
            field: "timeSec",
            type: "quantitative",
            title: "Step time (s)",
            format: ".2f",
          },
          {
            field: "efficiencyPct",
            type: "quantitative",
            title: "Efficiency (%)",
            format: ".1f",
          },
        ],
      },
    },
  ],
};

export const weakScalingMemorySpec = {
  $schema: "https://vega.github.io/schema/vega-lite/v6.json",
  description:
    "Peak per-GPU memory along each weak-scaling ladder, against the 32 GB card.",
  title: {
    text: "Constant work per GPU does not mean constant memory per GPU",
    subtitle:
      "Peak per-GPU memory along the same ladders. A flat line would mean the footprint is fully distributed.",
  },
  width: 620,
  height: 400,
  config,
  layer: [
    {
      mark: { type: "rule", strokeDash: [6, 5], color: guideColor },
      encoding: { y: { datum: 32 } },
    },
    {
      mark: {
        type: "text",
        align: "left",
        baseline: "bottom",
        dx: 6,
        dy: -4,
        color: annotationColor,
        fontSize: 12,
      },
      encoding: {
        y: { datum: 32 },
        x: { datum: 1 },
        text: { datum: "32 GB — V100 capacity" },
      },
    },
    {
      mark: lineMark,
      encoding: {
        x: gpuAxis,
        y: {
          field: "peakMemoryGb",
          type: "quantitative",
          title: "Peak memory per GPU (GB)",
          scale: { domain: [0, 32] },
        },
        color: workloadColor,
        tooltip: [
          {
            field: "localWorkloadMpix",
            type: "quantitative",
            title: "Mpix / GPU",
            format: ".2f",
          },
          { field: "imageSize", type: "nominal", title: "Image size" },
          { field: "gpuCount", type: "quantitative", title: "GPUs" },
          {
            field: "peakMemoryGb",
            type: "quantitative",
            title: "Peak memory (GB)",
            format: ".1f",
          },
        ],
      },
    },
  ],
};

export const checkpointingMemorySpec = {
  $schema: "https://vega.github.io/schema/vega-lite/v6.json",
  description:
    "Peak per-GPU memory with and without activation checkpointing, at max_batch_size 1.",
  title: {
    text: "Checkpointing is what keeps the run inside the card",
    subtitle:
      "Peak per-GPU memory at max_batch_size 1. Configurations marked below were not run: they do not fit in 32 GB.",
  },
  width: 560,
  height: 380,
  config,
  transform: [{ filter: "datum.maxBatchSize === 1" }],
  layer: [
    {
      transform: [{ filter: "datum.status === 'measured'" }],
      mark: { type: "bar", width: { band: 0.82 }, cornerRadiusEnd: 4 },
      encoding: {
        x: {
          field: "gpuCount",
          type: "ordinal",
          title: "GPUs",
          sort: "ascending",
        },
        xOffset: { field: "checkpointMode", sort: ["always", "never"] },
        y: {
          field: "peakMemoryMb",
          type: "quantitative",
          title: "Peak memory per GPU (MiB)",
          scale: { domain: [0, DEVICE_MEMORY_MB] },
        },
        color: checkpointColor,
        tooltip: [
          { field: "checkpointMode", type: "nominal", title: "Checkpointing" },
          { field: "gpuCount", type: "quantitative", title: "GPUs" },
          {
            field: "peakMemoryMb",
            type: "quantitative",
            title: "Peak memory (MiB)",
            format: ".0f",
          },
          {
            field: "forwardPeakMemoryMb",
            type: "quantitative",
            title: "Forward peak (MiB)",
            format: ".0f",
          },
          {
            field: "timeSec",
            type: "quantitative",
            title: "Step time (s)",
            format: ".3f",
          },
        ],
      },
    },
    {
      transform: [{ filter: "datum.status === 'measured'" }],
      mark: {
        type: "text",
        baseline: "bottom",
        dy: -5,
        fontSize: 12,
        color: annotationColor,
      },
      encoding: {
        x: { field: "gpuCount", type: "ordinal", sort: "ascending" },
        xOffset: { field: "checkpointMode", sort: ["always", "never"] },
        y: { field: "peakMemoryMb", type: "quantitative" },
        text: {
          field: "peakMemoryMb",
          type: "quantitative",
          format: ".0f",
        },
      },
    },
    {
      // Runs that are absent because they exceed the card, not because they
      // were skipped. Marked so the missing bar reads as a result.
      transform: [{ filter: "datum.status === 'did-not-fit'" }],
      mark: {
        type: "text",
        angle: 270,
        baseline: "middle",
        align: "left",
        fontSize: 12,
        fontStyle: "italic",
        color: annotationColor,
      },
      encoding: {
        x: { field: "gpuCount", type: "ordinal", sort: "ascending" },
        xOffset: { field: "checkpointMode", sort: ["always", "never"] },
        y: { datum: 1200 },
        text: { datum: "never — does not fit in 32 GB" },
      },
    },
    {
      mark: { type: "rule", strokeDash: [6, 5], color: guideColor },
      encoding: { y: { datum: DEVICE_MEMORY_MB } },
    },
  ],
};

// Five batch sizes is more than the three validated all-pairs slots, so this
// ordinal ramp carries the ordered magnitude instead of five categorical hues.
const batchRamp = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"];

const batchColor = {
  field: "maxBatchSize",
  type: "ordinal",
  title: "max_batch_size",
  scale: { range: batchRamp },
};

/**
 * Efficiency against GPU count, one line per batch size, all sharing a single
 * baseline (largest batch at the fewest GPUs) so the curves are comparable in
 * absolute terms — not normalized per GPU count, which would pin every line to
 * the same start and erase the batch-size effect this figure is about.
 */
export const batchSizeEfficiencySpec = {
  $schema: "https://vega.github.io/schema/vega-lite/v6.json",
  description:
    "Parallel efficiency against GPU count, one line per max_batch_size, against a single shared baseline.",
  title: {
    text: "Larger batches are faster at every GPU count",
    subtitle:
      "4096x4096, checkpointing on. All series share one baseline: max_batch_size 16 on 1 GPU.",
  },
  width: 620,
  height: 400,
  config,
  layer: [
    ...idealRule(100, "baseline: batch 16 on 1 GPU"),
    {
      mark: lineMark,
      encoding: {
        x: {
          field: "gpuCount",
          type: "quantitative",
          title: "GPUs",
          scale: { type: "log", base: 2 },
          axis: { values: [1, 4, 16], format: "d" },
        },
        y: {
          field: "efficiencyPct",
          type: "quantitative",
          title: "Parallel efficiency (%)",
          scale: { domain: [0, 110] },
        },
        color: batchColor,
        tooltip: [
          { field: "maxBatchSize", type: "quantitative", title: "Max batch" },
          { field: "gpuCount", type: "quantitative", title: "GPUs" },
          {
            field: "timeSec",
            type: "quantitative",
            title: "Step time (s)",
            format: ".2f",
          },
          {
            field: "efficiencyPct",
            type: "quantitative",
            title: "Efficiency (%)",
            format: ".1f",
          },
          {
            field: "peakMemoryGb",
            type: "quantitative",
            title: "Peak memory (GB)",
            format: ".1f",
          },
        ],
      },
    },
  ],
};

/** Absolute peak memory, so the price of each batch size is readable directly. */
export const batchSizeMemorySpec = {
  $schema: "https://vega.github.io/schema/vega-lite/v6.json",
  description: "Peak per-GPU memory by max_batch_size and GPU count.",
  title: {
    text: "And the memory they cost, in absolute terms",
    subtitle:
      "Peak per-GPU memory for the same runs. The dashed line is the 32 GB V100 capacity.",
  },
  width: 620,
  height: 380,
  config,
  layer: [
    {
      mark: { type: "bar", width: { band: 0.82 }, cornerRadiusEnd: 4 },
      encoding: {
        x: {
          field: "maxBatchSize",
          type: "ordinal",
          title: "max_batch_size",
          sort: "ascending",
          axis: { labelAngle: 0 },
        },
        xOffset: { field: "gpuCount", sort: "ascending" },
        y: {
          field: "peakMemoryGb",
          type: "quantitative",
          title: "Peak memory per GPU (GB)",
          scale: { domain: [0, 32] },
        },
        color: {
          field: "gpuCount",
          type: "ordinal",
          title: "GPUs",
          scale: { range: series },
        },
        tooltip: [
          { field: "gpuCount", type: "quantitative", title: "GPUs" },
          { field: "maxBatchSize", type: "quantitative", title: "Max batch" },
          {
            field: "peakMemoryGb",
            type: "quantitative",
            title: "Peak memory (GB)",
            format: ".1f",
          },
          {
            field: "timeSec",
            type: "quantitative",
            title: "Step time (s)",
            format: ".2f",
          },
        ],
      },
    },
    {
      mark: {
        type: "text",
        baseline: "bottom",
        dy: -4,
        fontSize: 10,
        color: annotationColor,
      },
      encoding: {
        x: { field: "maxBatchSize", type: "ordinal", sort: "ascending" },
        xOffset: { field: "gpuCount", sort: "ascending" },
        y: { field: "peakMemoryGb", type: "quantitative" },
        text: { field: "peakMemoryGb", type: "quantitative", format: ".1f" },
      },
    },
    {
      mark: { type: "rule", strokeDash: [6, 5], color: guideColor },
      encoding: { y: { datum: 32 } },
    },
  ],
};
