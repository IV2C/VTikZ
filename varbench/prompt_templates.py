SYSTEM_PROMPT_GENERATION: str = """
You are an expert coding assistant specialized in modifying file contents based on instructions.
Given an instruction and file content, respond only with the updated file's full content, ensuring it is entirely enclosed between code tags like this
```tex
content
```

Provide no additional text or explanations beyond the code tags.
"""


SYSTEM_PROMPT_INSTRUCTIONS = """
You are an expert in creating and editing code.
You will receive code that generates an image (using formats like TikZ, SVG, ASCII art, etc.). 
You will also be asked to comment the code in order to have a full understanding of it, i.e. give the full code given as input, with comments that identify the features in the code. For instance, with a cat's face as input, you comment which part of the code makes which part of the image.
Your task is to provide examples of modifications to this code.
For instance, if given code that creates an image of a white cat's face, you could provide an instruction like "change the cat to gray," a result description such as "a gray cat".
Provide high level, inventive, yet doable modifications that necessitates little modification.
The modifications that you provide could be ones asked by a someone who does not want to see the code but wants to make an image.
In the case of a cat, some great examples could be "make the cat's ears bigger","change the colors of the cat's cheeks to purple"
Generate {number_generation} different examples of modifications.
"""


IT_PROMPT: str = """
{instruction}
```tex
{content}
```
"""

PROMPT: str = """
<instruction>
{instruction}
</instruction>
<content>
{content}
</content>
<answer>
Here is the new content updated according to your instruction
```
"""
