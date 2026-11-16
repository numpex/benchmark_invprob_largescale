// Vega-Lite specs for the cryo-ET training finding (Adastra MI300A).
import {
  annotationColor,
  config,
  guideColor,
  idealRule,
  lineMark,
  series,
} from "./trainingScalingCharts";

const schema = "https://vega.github.io/schema/vega-lite/v6.json";

/** Per-GPU memory of an MI300A, in GB. */
export const DEVICE_MEMORY_GB = 128;

const gpuAxis = {
  field: "gpuCount",
  type: "quantitative",
  title: "GPUs",
  scale: { type: "log", base: 2 },
  axis: { values: [4, 8, 16, 32, 64, 128], format: "d" },
};

const efficiencyAxis = {
  field: "efficiencyPct",
  type: "quantitative",
  title: "Parallel efficiency (%)",
  scale: { domain: [0, 110] },
};

const measured = { filter: "datum.status === 'measured'" };

const timeTooltip = [
  {
    field: "trainSec",
    type: "quantitative",
    title: "Fwd + bwd (s)",
    format: ".1f",
  },
  {
    field: "peakMemoryGb",
    type: "quantitative",
    title: "Peak memory (GB)",
    format: ".1f",
  },
];

export const speedupSpec = {
  $schema: schema,
  title: {
    text: "bf16 carries the speedup",
    subtitle:
      "Forward + backward time on 4 GPUs, one change at a time against fp32.",
  },
  width: 560,
  height: 260,
  config,
  encoding: {
    y: {
      field: "variant",
      type: "nominal",
      title: null,
      sort: [
        "baseline",
        "compile",
        "distributed physics",
        "bf16",
        "all combined",
      ],
    },
    x: {
      field: "trainSec",
      type: "quantitative",
      title: "Forward + backward (s)",
    },
  },
  layer: [
    {
      mark: { type: "bar", color: series[0], cornerRadiusEnd: 4 },
      encoding: {
        tooltip: [
          { field: "variant", type: "nominal", title: "Variant" },
          ...timeTooltip,
          {
            field: "speedupPct",
            type: "quantitative",
            title: "Speedup (%)",
            format: "+.1f",
          },
        ],
      },
    },
    {
      // Speedup over the baseline in percent: (t0 / t - 1) x 100.
      transform: [
        {
          calculate:
            "format(datum.speedupPct, datum.speedupPct >= 10 ? '+.0f' : '+.1f') + '%'",
          as: "label",
        },
      ],
      mark: {
        type: "text",
        align: "left",
        dx: 5,
        fontSize: 12,
        color: annotationColor,
      },
      encoding: { text: { field: "label", type: "nominal" } },
    },
  ],
};

/** One panel of the patch-size figure: a bar per patch size for `field`. */
// Set per layer, so the 128 GB rule does not inherit it.
const patchAxis = {
  field: "patchSize",
  type: "ordinal",
  title: "Patch size (voxels per side)",
  axis: { labelAngle: 0 },
};

const patchPanel = (
  field: string,
  title: string,
  format: string,
  extra: object[] = [],
) => ({
  width: 300,
  height: 280,
  layer: [
    {
      transform: [measured],
      mark: { type: "bar", color: series[0], cornerRadiusEnd: 4 },
      encoding: {
        x: patchAxis,
        y: { field, type: "quantitative", title },
        tooltip: [
          { field: "patchSize", type: "ordinal", title: "Patch" },
          ...timeTooltip,
        ],
      },
    },
    {
      transform: [measured],
      mark: {
        type: "text",
        baseline: "bottom",
        dy: -5,
        fontSize: 12,
        color: annotationColor,
      },
      encoding: {
        x: patchAxis,
        y: { field, type: "quantitative" },
        text: { field, type: "quantitative", format },
      },
    },
    {
      transform: [{ filter: "datum.status !== 'measured'" }],
      mark: {
        type: "text",
        angle: 270,
        align: "left",
        fontSize: 12,
        fontStyle: "italic",
        color: annotationColor,
      },
      encoding: {
        x: patchAxis,
        y: { datum: 0 },
        dy: { value: -4 },
        text: { field: "status" },
      },
    },
    ...extra,
  ],
});

export const patchSizeSpec = {
  $schema: schema,
  title: {
    text: "Larger patches are faster, up to a hard limit",
    subtitle:
      "Forward + backward time and peak memory per GPU, 4 GPUs, bf16, compiled.",
  },
  config,
  hconcat: [
    patchPanel("trainSec", "Forward + backward (s)", ".0f"),
    patchPanel("peakMemoryGb", "Peak memory per GPU (GB)", ".1f", [
      {
        mark: { type: "rule", strokeDash: [6, 5], color: guideColor },
        encoding: { y: { datum: DEVICE_MEMORY_GB } },
      },
    ]),
  ],
};

const checkpointColor = {
  field: "checkpointMode",
  type: "nominal",
  title: "Checkpointing",
  scale: { domain: ["always", "never"], range: [series[0], series[1]] },
};

export const checkpointingEfficiencySpec = {
  $schema: schema,
  title: {
    text: "Efficiency drops as the volume spreads across nodes",
    subtitle: "Parallel efficiency of forward + backward against 4 GPUs.",
  },
  width: 560,
  height: 340,
  config,
  transform: [measured],
  layer: [
    ...idealRule(100, "perfect scaling"),
    {
      mark: lineMark,
      encoding: {
        x: gpuAxis,
        y: efficiencyAxis,
        color: checkpointColor,
        tooltip: [
          { field: "checkpointMode", type: "nominal", title: "Checkpointing" },
          { field: "gpuCount", type: "quantitative", title: "GPUs" },
          ...timeTooltip,
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

// Bars per GPU count, one offset per checkpointing mode.
const gpuBands = {
  x: {
    field: "gpuCount",
    type: "ordinal",
    title: "GPUs",
    sort: "ascending",
    axis: { labelAngle: 0 },
  },
  xOffset: { field: "checkpointMode", sort: ["always", "never"] },
};

export const checkpointingMemorySpec = {
  $schema: schema,
  title: {
    text: "Without checkpointing the volume does not fit",
    subtitle: "Peak per-GPU memory. Marked runs ran out of the 128 GB.",
  },
  width: 560,
  height: 340,
  config,
  layer: [
    {
      transform: [measured],
      mark: { type: "bar", cornerRadiusEnd: 4 },
      encoding: {
        ...gpuBands,
        y: {
          field: "peakMemoryGb",
          type: "quantitative",
          title: "Peak memory per GPU (GB)",
          scale: { domain: [0, DEVICE_MEMORY_GB] },
        },
        color: checkpointColor,
        tooltip: [
          { field: "checkpointMode", type: "nominal", title: "Checkpointing" },
          ...timeTooltip,
        ],
      },
    },
    {
      transform: [measured],
      mark: {
        type: "text",
        baseline: "bottom",
        dy: -4,
        fontSize: 11,
        color: annotationColor,
      },
      encoding: {
        ...gpuBands,
        y: { field: "peakMemoryGb", type: "quantitative" },
        text: { field: "peakMemoryGb", type: "quantitative", format: ".0f" },
      },
    },
    {
      transform: [{ filter: "datum.status === 'out of memory'" }],
      mark: {
        type: "text",
        angle: 270,
        align: "left",
        fontSize: 12,
        fontStyle: "italic",
        color: annotationColor,
      },
      encoding: {
        ...gpuBands,
        y: { datum: 4 },
        text: { datum: "never: out of memory" },
      },
    },
    {
      mark: { type: "rule", strokeDash: [6, 5], color: guideColor },
      encoding: { y: { datum: DEVICE_MEMORY_GB } },
    },
  ],
};

// One distinct hue per number of GPUs per volume.
const tilingGroups = ["1", "2", "4", "8", "16", "32", "64"];
const tilingColor = {
  field: "group",
  type: "nominal",
  title: "GPUs per volume",
  scale: {
    domain: tilingGroups,
    range: [
      "#1baf7a",
      "#2a78d6",
      "#8b5cf6",
      "#d4a017",
      "#d6457f",
      "#8c564b",
      series[1],
    ],
  },
};

const tilingLine = (y: object) => ({
  mark: lineMark,
  encoding: {
    x: { ...gpuAxis, axis: { values: [4, 16, 32, 64], format: "d" } },
    y,
    color: tilingColor,
    tooltip: [
      { field: "gpuCount", type: "quantitative", title: "GPUs" },
      {
        field: "gpusPerVolume",
        type: "quantitative",
        title: "GPUs per volume",
      },
      {
        field: "perVolumeSec",
        type: "quantitative",
        title: "Time per volume (s)",
        format: ".1f",
      },
      {
        field: "efficiencyPct",
        type: "quantitative",
        title: "Efficiency (%)",
        format: ".1f",
      },
    ],
  },
});

// Full range on the left; 1, 2 and 4 GPUs per volume zoomed on the right.
export const tilingSpec = {
  $schema: schema,
  title: {
    text: "Small tiling groups plus DDP keep the GPUs busy",
    subtitle:
      "Efficiency per volume against 4 GPUs. Right: 1, 2 and 4 GPUs per volume, zoomed.",
  },
  config,
  transform: [{ calculate: "'' + datum.gpusPerVolume", as: "group" }],
  resolve: { scale: { y: "independent" } },
  hconcat: [
    {
      width: 340,
      height: 340,
      layer: [...idealRule(100, "perfect scaling"), tilingLine(efficiencyAxis)],
    },
    {
      width: 220,
      height: 340,
      transform: [{ filter: "datum.gpusPerVolume <= 4" }],
      layer: [
        ...idealRule(100, ""),
        tilingLine({
          ...efficiencyAxis,
          scale: { domain: [94, 102], zero: false },
        }),
      ],
    },
  ],
};
