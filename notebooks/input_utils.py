import re
from datasets import Dataset
import matplotlib.pyplot as plt
import seaborn as sns


class SvgUtils:
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
    class Filters:

        def noAnimations(row):
            return all(
                anim_tag not in row["Svg"]
                for anim_tag in [
                    "<animate ",
                    "<animateMotion ",
                    "<animateTransform ",
                    "<set ",
                ]
            )

        def noText(row):
            return all(text_tag not in row["Svg"] for text_tag in ["<text>", "<tspan>"])

        def hasBasicShape(row):
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

        def maxCharacters(row):
            return  len(row["Svg"]) < 4000 and len(row["Svg"]) > 500

import numpy as np

class TikzUtils:
    def statistics(dataset: Dataset):
        dataset = dataset.map(
            lambda row: {"num_characters": len(row["code"]),"num_lines": len(row["code"].splitlines())},
            desc="computing length of the code",
        )
        df = dataset.to_pandas()

        # Compute IQR
        Q1 = np.percentile(df["num_characters"], 25)
        Q3 = np.percentile(df["num_characters"], 75)
        IQR = Q3 - Q1

        # Define outlier bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        print(lower_bound)
        print(upper_bound)
        
        # Filter out outliers
        df_filtered = df[(df["num_characters"] >= lower_bound) & (df["num_characters"] <= upper_bound)]

        plt.figure(figsize=(12, 6))
        sns.histplot(x=df_filtered["num_characters"], color="lightgreen")

        # Set title and labels
        plt.title('Histogram of Number of Characters (Outliers Removed)')
        plt.xlabel("Number of Characters")
        plt.show()
        
        sns.histplot(x=df_filtered["num_lines"], color="lightgreen")

        # Set title and labels
        plt.title('Histogram of number of lines')
        plt.xlabel("Lines")
        plt.show()
        
        sns.histplot(y=df_filtered["origin"], color="lightgreen")

        # Set title and labels
        plt.title('Histogram of number of origins')
        plt.ylabel("Origin")
        plt.show()
        

    class Filters:
        def characterLength(row):
            return  len(row["code"]) < 3570 and len(row["code"]) > 700
        def lineLength(row):
            code_line_length = row["code"].splitlines()
            return  len(code_line_length) > 50 and len(code_line_length) < 80    
        def noLLMorigin(row):
            return row["origin"] != "gpt4" and row["origin"] != "chatgpt"
        def hasBasicShape(row):
            return sum(
                row["code"].count(shape_tag)
                for shape_tag in [
                    "\\fill",
                    "\\draw"
                ]
            ) > 3
        def oneTikzPicture(row):
            return row["code"].count("\\begin{tikzpicture}") == 1
        
        def hasComments(row):
            return bool(re.search(r'(?<!\\)%[^\s%].*', row["code"]))
    class Modifications:
        def removeComments(row):
            row["full_code"] = row["code"]
            row["code"] = "\n".join([re.split(r'(?<!\\)%', line)[0] for line in row["code"].splitlines()])
            return row
