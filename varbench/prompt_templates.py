# basic llm
SYSTEM_PROMPT_GENERATION: str = """
You are an expert coding assistant specialized in modifying file contents based on instructions.
Given an instruction and file content, respond only with the updated file's full content, ensuring it is entirely enclosed between code tags like this
```
content
```

Provide no additional text or explanations beyond the code tags.
"""
# basic LMM
MULTIMODAL_INSTRUCTION: str = """
You are an expert coding assistant specialized in modifying file contents based on instructions.
Given an instruction, file content and the image that the current file creates, respond only with the updated file's full content, ensuring it is entirely enclosed between code tags like this
```
content
```

Provide no additional text or explanations beyond the code tags.

Here is the instruction:
{instruction}
```
{content}
```
"""
## simple prompt
IT_PROMPT: str = """
{instruction}
```
{content}
```
"""
## loop vlm
VLM_INSTRUCTION: str = """
Detail every feature of the image precisely
"""
SYSTEM_PROMPT_GENERATION_VLM_LOOP: str = """
You are an expert coding assistant specialized in modifying file contents based on instructions.
First, given an instruction and file content, respond with the updated full file content.
Then, you will be given a description of the image you have created, and it is up to you to decide wether or not you have made the right changes, and apply new ones or not otherwise.
For each response, respond only with the file's full content, ensuring it is entirely enclosed between code tags like this.
```
content
```

Provide no additional text or explanations beyond the code tags.
"""


## multimodal loop
MULTIMODAL_VISION_DESCRIPTION_INSTRUCTION: str = """
I gave to a system these instruction to make changes to an existing image:
{instruction}

If the system made the right changes, answer with the exact words "The changes satisfy the instructions".
Otherwise, give me a detailed explanation of the reasons why the changes made to the image do not satisfy the instructions.
"""

MULTIMODAL_LOOP_SYSTEM_INSTRUCTION:str = """
You are a coding assistant specializing in modifying file content to achieve specific visual results. When given an instruction and a code file, update the file based on the instruction and respond with the updated file's full content enclosed in triple backticks (` ``` `).
Afterward, you will be shown the resulting image. Describe how your modifications affected the image and evaluate how closely the results align with the instruction. Always seek opportunities to improve the output by refining the code and provide a revised version, even if the changes are minor. Only respond with "instruction satisfied" if no meaningful adjustments can further enhance the result.
"""

MULTIMODAL_LOOP_INSTRUCTION: str = """
Here is the image that the code generated creates. 
"""

##synthetic data generation
SYSTEM_PROMPT_INSTRUCTIONS_SYNTHETIC = """
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


#Code corrector agent
CODE_CORRECTOR_SYSTEM_PROMPT:str = """
Correct the given code based on the provided compiler error, then present the entire corrected version.

# Steps

1. Analyze the provided code and the compiler error message.
2. Identify the root cause of the error.
3. Correct the code to resolve the error while maintaining the intended functionality.
4. Provide the entire corrected code using markdown code blocks.

# Output Format

Provide the entire corrected version of the code enclosed in a markdown code block:

```
[Corrected Code]
```
"""

CODE_CORRECTOR_IT_PROMPT:str = """
{error_message}

{code}
"""