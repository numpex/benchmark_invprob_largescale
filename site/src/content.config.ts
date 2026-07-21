import { defineCollection } from "astro:content";
import { glob } from "astro/loaders";
import { z } from "astro/zod";

const technicalContent = defineCollection({
  loader: glob({
    base: "./src/pages/technical_content",
    pattern: "**/*.{md,mdx}",
  }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    eyebrow: z.string(),
    publishedAt: z.coerce.date().optional(),
    readingTime: z.string().optional(),
    featured: z.boolean().default(false),
    pageType: z.enum(["article", "listing"]).default("article"),
  }),
});

export const collections = { technicalContent };
