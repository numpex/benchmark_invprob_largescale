const colors = ["#168d80", "#f2b46d", "#50617d"];

const config = {
  background: null,
  font: "Inter, ui-sans-serif, system-ui, sans-serif",
  view: { stroke: null },
  axis: {
    domainColor: "#8290a8",
    gridColor: "#e5e8e2",
    labelColor: "#50617d",
    titleColor: "#20304e",
    titleFontWeight: 600,
  },
  legend: {
    labelColor: "#50617d",
    titleColor: "#20304e",
    orient: "bottom",
    direction: "horizontal",
  },
};

const gpuAxis = {
  field: "gpuCount",
  type: "quantitative",
  title: "GPUs",
  scale: { type: "log", base: 2 },
  axis: { values: [1, 2, 4, 8, 16, 32, 64], format: "d" },
};

export const timingBreakdownSpec = {
  $schema: "https://vega.github.io/schema/vega-lite/v6.json",
  description:
    "Inference iteration time split into physics, denoising, and residual overhead.",
  title: {
    text: "Timing breakdown",
    subtitle:
      "Iterations 1–2 excluded; averages use iterations 3–10. One-GPU 2048²/4096² is non-tiled; distributed points include halo work.",
  },
  config,
  facet: {
    column: {
      field: "imageSize",
      type: "ordinal",
      title: "Image side (pixels)",
      header: { labelFontSize: 12, titleFontSize: 12 },
    },
  },
  spec: {
    width: 210,
    height: 245,
    transform: [
      {
        fold: ["physicsSec", "denoisingSec", "overheadSec"],
        as: ["component", "seconds"],
      },
    ],
    mark: { type: "area", line: true, point: { size: 36 }, opacity: 0.72 },
    encoding: {
      x: gpuAxis,
      y: {
        field: "seconds",
        type: "quantitative",
        stack: "zero",
        title: "Mean time per iteration (s)",
      },
      color: {
        field: "component",
        type: "nominal",
        title: null,
        scale: {
          domain: ["physicsSec", "denoisingSec", "overheadSec"],
          range: colors,
        },
        legend: {
          labelExpr:
            "datum.label === 'physicsSec' ? 'Physics' : datum.label === 'denoisingSec' ? 'Denoising' : 'Residual overhead'",
        },
      },
      tooltip: [
        { field: "imageSize", type: "ordinal", title: "Image side" },
        { field: "gpuCount", type: "quantitative", title: "GPUs" },
        { field: "mode", type: "nominal", title: "Execution" },
        {
          field: "workMultiplier",
          type: "quantitative",
          title: "Tile work multiplier",
          format: ".3f",
        },
        { field: "component", type: "nominal", title: "Component" },
        {
          field: "seconds",
          type: "quantitative",
          title: "Seconds",
          format: ".4f",
        },
      ],
    },
  },
  resolve: { scale: { y: "independent" } },
};

export const qualityPreservationSpec = {
  $schema: "https://vega.github.io/schema/vega-lite/v6.json",
  description:
    "PSNR difference from the single-GPU configuration at each iteration.",
  title: {
    text: "Reconstruction quality is preserved",
    subtitle:
      "Signed PSNR difference at the same iteration; 2048² and 4096² use their single-GPU run as reference.",
  },
  config,
  facet: {
    column: {
      field: "imageSize",
      type: "ordinal",
      title: "Image side (pixels)",
      header: { labelFontSize: 12, titleFontSize: 12 },
    },
  },
  spec: {
    width: 210,
    height: 245,
    layer: [
      {
        mark: { type: "rule", strokeDash: [6, 5], color: "#8290a8" },
        encoding: { y: { datum: 0 } },
      },
      {
        mark: {
          type: "line",
          point: { filled: true, size: 38 },
          strokeWidth: 2.1,
        },
        encoding: {
          x: {
            field: "iteration",
            type: "quantitative",
            title: "PnP iteration",
            axis: { values: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] },
          },
          y: {
            field: "psnrDifferenceDb",
            type: "quantitative",
            title: "ΔPSNR from reference (dB)",
            scale: { domain: [-0.01, 0.01] },
            axis: { format: ".4f" },
          },
          color: {
            field: "gpuCount",
            type: "nominal",
            title: "GPUs",
            scale: { scheme: "viridis" },
          },
          tooltip: [
            { field: "imageSize", type: "ordinal", title: "Image side" },
            { field: "iteration", type: "quantitative", title: "Iteration" },
            { field: "gpuCount", type: "quantitative", title: "GPUs" },
            {
              field: "baselineGpuCount",
              type: "quantitative",
              title: "Reference GPUs",
            },
            { field: "psnrDb", type: "quantitative", title: "PSNR (dB)", format: ".6f" },
            {
              field: "psnrDifferenceDb",
              type: "quantitative",
              title: "ΔPSNR (dB)",
              format: ".6f",
            },
          ],
        },
      },
    ],
  },
};

export const efficiencySpec = (
  field: "absoluteEfficiencyPct" | "workNormalizedEfficiencyPct",
  yTitle: string,
) => ({
  $schema: "https://vega.github.io/schema/vega-lite/v6.json",
  description: yTitle,
  title: {
    text: yTitle,
    subtitle:
      "Iterations 1–2 excluded; averages start at iteration 3 (iterations 3–10).",
  },
  width: 680,
  height: 360,
  config,
  layer: [
    {
      mark: { type: "rule", strokeDash: [6, 5], color: "#8290a8" },
      encoding: { y: { datum: 100 } },
    },
    {
      mark: {
        type: "line",
        point: { filled: true, size: 66 },
        strokeWidth: 2.5,
      },
      encoding: {
        x: gpuAxis,
        y: {
          field,
          type: "quantitative",
          title: yTitle,
          scale: { domainMin: 0 },
        },
        color: {
          field: "imageSize",
          type: "nominal",
          title: "Image side",
          scale: { range: colors },
        },
        tooltip: [
          { field: "imageSize", type: "nominal", title: "Image side" },
          { field: "gpuCount", type: "quantitative", title: "GPUs" },
          { field, type: "quantitative", title: yTitle, format: ".1f" },
          {
            field: "totalSec",
            type: "quantitative",
            title: "Iteration (s)",
            format: ".4f",
          },
          {
            field: "workMultiplier",
            type: "quantitative",
            title: "Tile work multiplier",
            format: ".3f",
          },
        ],
      },
    },
  ],
});

export const communicationScalingSpec = {
  $schema: "https://vega.github.io/schema/vega-lite/v6.json",
  description:
    "Compute and attributed communication scaling for 2D and 3D distributed PnP.",
  title: {
    text: "Compute and communication scaling",
    subtitle:
      "Iterations 1–2 excluded; averages start at iteration 3 (2D: 3–10, 3D: 3–5).",
  },
  config,
  vconcat: [
    {
      width: 680,
      height: 260,
      transform: [{ filter: "datum.mode === 'distributed'" }],
      layer: [
        {
          mark: { type: "line", strokeDash: [6, 5], color: "#8290a8" },
          encoding: {
            x: gpuAxis,
            y: {
              field: "idealComputeSpeedup",
              type: "quantitative",
              title: "Compute speedup",
            },
            detail: { field: "problem" },
          },
        },
        {
          mark: {
            type: "line",
            point: { filled: true, size: 58 },
            strokeWidth: 2.5,
          },
          encoding: {
            x: gpuAxis,
            y: {
              field: "computeSpeedup",
              type: "quantitative",
              title: "Compute speedup",
            },
            color: {
              field: "problem",
              type: "nominal",
              title: null,
              scale: { range: colors },
            },
            tooltip: [
              { field: "problem", type: "nominal", title: "Problem" },
              { field: "gpuCount", type: "quantitative", title: "GPUs" },
              {
                field: "computeSpeedup",
                type: "quantitative",
                title: "Compute speedup",
                format: ".2f",
              },
              { field: "nodeCount", type: "quantitative", title: "Nodes" },
            ],
          },
        },
      ],
    },
    {
      width: 680,
      height: 260,
      transform: [
        { filter: "datum.mode === 'distributed'" },
        {
          fold: ["computeCudaSec", "communicationCudaSec"],
          as: ["component", "seconds"],
        },
        { filter: "datum.seconds > 0" },
      ],
      mark: {
        type: "line",
        point: { filled: true, size: 52 },
        strokeWidth: 2.3,
      },
      encoding: {
        x: gpuAxis,
        y: {
          field: "seconds",
          type: "quantitative",
          title: "CUDA time (s, log scale)",
          scale: { type: "log" },
        },
        color: {
          field: "problem",
          type: "nominal",
          title: "Problem",
          scale: { range: colors },
        },
        strokeDash: {
          field: "component",
          type: "nominal",
          title: "Profiler component",
          scale: {
            domain: ["computeCudaSec", "communicationCudaSec"],
            range: [
              [1, 0],
              [7, 5],
            ],
          },
          legend: {
            labelExpr:
              "datum.label === 'computeCudaSec' ? 'Compute' : 'Communication'",
            symbolType: "stroke",
            symbolSize: 180,
            symbolStrokeWidth: 3,
            symbolStrokeColor: "#20304e",
            symbolFillColor: "transparent",
          },
        },
        tooltip: [
          { field: "problem", type: "nominal", title: "Problem" },
          { field: "gpuCount", type: "quantitative", title: "GPUs" },
          { field: "component", type: "nominal", title: "Component" },
          {
            field: "seconds",
            type: "quantitative",
            title: "Seconds",
            format: ".4f",
          },
          {
            field: "communicationSharePct",
            type: "quantitative",
            title: "Communication share (%)",
            format: ".1f",
          },
        ],
      },
    },
  ],
  resolve: { scale: { color: "shared" } },
};
