import type { VariantProps } from "class-variance-authority";
import { cva } from "class-variance-authority";

export { default as SegmentedControl } from "./SegmentedControl.vue";
export { default as SegmentedControlItem } from "./SegmentedControlItem.vue";

export const segmentedControlVariants = cva(
  "flex max-w-full shrink-0 flex-wrap items-center rounded-lg",
  {
    variants: {
      variant: {
        raised: "border border-border/55 bg-muted/60 p-[3px]",
        quiet: "p-0",
      },
      fill: {
        true: "w-full",
        false: "w-fit",
      },
    },
    defaultVariants: {
      variant: "raised",
      fill: false,
    },
  }
);

export const segmentedControlItemVariants = cva(
  "inline-flex items-center justify-center gap-1.5 whitespace-nowrap rounded-md border text-xs transition-all duration-200 outline-none focus-visible:ring-2 focus-visible:ring-ring/50 disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-3.5",
  {
    variants: {
      size: {
        sm: "touch-target px-2.5 py-1 text-[11px] max-md:min-h-11 max-md:px-3",
        md: "touch-target px-3 py-1.5 text-xs max-md:min-h-11 max-md:px-4",
        // 32px items + 6px frame padding + 2px border = the 40px control metric
        // a toolbar band uses, so an icon switch sits on the same baseline as
        // the search field beside it. `md` stays 30px: its tab strips live in
        // page headers, where the metric is 36px.
        icon: "touch-target touch-target-square size-8 p-0 max-md:size-11",
      },
      variant: {
        raised: "",
        quiet: "",
      },
      active: {
        true: "font-semibold",
        false: "font-medium",
      },
      fill: {
        true: "flex-1",
        false: "",
      },
      caps: {
        true: "uppercase tracking-wider",
        false: "",
      },
    },
    compoundVariants: [
      {
        fill: false,
        class: "shrink-0",
      },
      {
        variant: "raised",
        active: true,
        class: "border-border/70 bg-accent text-foreground shadow-xs",
      },
      {
        variant: "raised",
        active: false,
        class: "border-transparent text-muted-foreground hover:bg-accent/40 hover:text-foreground",
      },
      {
        variant: "quiet",
        active: true,
        class: "border-transparent bg-muted text-foreground",
      },
      {
        variant: "quiet",
        active: false,
        class: "border-transparent text-muted-foreground hover:bg-muted/60 hover:text-foreground",
      },
    ],
    defaultVariants: {
      size: "md",
      variant: "raised",
      active: false,
      fill: false,
      caps: false,
    },
  }
);

export type SegmentedControlVariants = VariantProps<typeof segmentedControlVariants>;
export type SegmentedControlItemVariants = VariantProps<typeof segmentedControlItemVariants>;
