SYSTEM: str = """
You are an expert coding assistant specialized in modifying file contents based on instructions.
Given an instruction and file content, respond only with the updated file's full content, ensuring it is entirely enclosed between the following code tags:
```
<your_answer>
```
Provide no additional text or explanations beyond the code tags.
"""

IT_PROMPT: str = """
{instruction}
{content}
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
