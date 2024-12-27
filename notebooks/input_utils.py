import re
from datasets import Dataset
import matplotlib.pyplot as plt
import seaborn as sns


class SvgFilters:

    def statistics(dataset: Dataset):
        tags = [
            "<circle ",
            "<rect ",
            "<ellipse ",
            "<polygon ",
            "<polyline ",
            "<line ",
            "<animate ",
            "<animateMotion ",
            "<animateTransform ",
            "<set ",
        ]

        # Could be optimized by doing everything in one dataset.map ?

        for tag in tags:
            dataset = dataset.map(
                lambda row: {tag: row["Svg"].count(tag)},
                desc=f"Computing number of {tag} in the svg",
            )

        dataset = dataset.map(
            lambda row: {"num_characters": len(row["Svg"])},
            desc="computing length of the svg",
        )
        df = dataset.to_pandas()

        plt.figure(figsize=(12, 6))
        sns.boxplot(x=df["num_characters"], color="lightgreen")

        # Set title and labels
        plt.title('Boxplot of Number of Characters in the "Svg" Column')
        plt.xlabel("Number of Characters")
        plt.show()

        # Create subplots
        fig, axes = plt.subplots(len(tags), 1, figsize=(12, 4 * len(tags)))

        # Ensure axes is iterable even when there is just one tag
        if len(tags) == 1:
            axes = [axes]

        # Plot each tag
        for i, tag in enumerate(tags):
            sns.histplot(df[tag], bins=30, kde=False, color="skyblue", ax=axes[i])

            # Set log scale and labels
            axes[i].set_yscale("log")
            axes[i].set_title(f"Distribution of {tag} Counts (Log Scale)")
            axes[i].set_xlabel(f"{tag} Count")
            axes[i].set_ylabel("Frequency (Log Scale)")

        plt.tight_layout()
        plt.show()

    def noAnimations(row: str):
        return all(
            anim_tag not in row["Svg"]
            for anim_tag in [
                "<animate ",
                "<animateMotion ",
                "<animateTransform ",
                "<set ",
            ]
        )

    def noText(row: str):
        return all(text_tag not in row["Svg"] for text_tag in ["<text>", "<tspan>"])

    def hasBasicShape(row: str):
        return any(
            shape_tag in row["Svg"]
            for shape_tag in [
                "<circle ",
                "<rect ",
                "<ellipse ",
                "<polygon ",
                "<polyline ",
                "<line ",
            ]
        )

    def maxCharacters(row: str):
        return  len(row["Svg"]) < 4000 and len(row["Svg"]) > 500


class TikzFilters:
