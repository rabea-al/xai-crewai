from xai_components.base import InArg, OutArg, Component, xai_component
from crewai import Agent, Crew, Task, LLM
from langchain_openai import ChatOpenAI
import os
import json
from PIL import Image
from langchain_community.tools import tool
from dotenv import load_dotenv

load_dotenv()

@xai_component
class CrewAIMakeToolbelt(Component):
    name: InArg[str]
    toolbelt_spec: OutArg[list]

    def execute(self, ctx) -> None:
        toolbelt_name = self.name.value if self.name.value else "default"
        toolbelt_key = f"toolbelt_{toolbelt_name}"

        if toolbelt_key in ctx:
            self.toolbelt_spec.value = list(ctx[toolbelt_key].values())
        else:
            self.toolbelt_spec.value = []

@xai_component
class CrewAIInit(Component):
    agent_name: InArg[str]
    role: InArg[str]
    goal: InArg[str]
    backstory: InArg[str]
    toolbelt_spec: InArg[list]
    agent: OutArg[Agent]

    def execute(self, ctx) -> None:
        chat_llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7)

        self.agent.value = Agent(
            role=self.role.value,
            goal=self.goal.value,
            backstory=self.backstory.value if self.backstory.value else "No backstory provided.",
            tools=self.toolbelt_spec.value,
            llm=chat_llm
        )


@xai_component
class CrewAIRunTasks(Component):
    """
    Runs tasks using the provided CrewAI agent.

    ##### inPorts:
    - agent: The agent responsible for executing tasks.
    - task_description: The description of the task to execute.

    ##### outPorts:
    - result: The output result of the task execution.
    """

    agent: InArg[Agent]
    task_description: InArg[str]
    result: OutArg[str]

    def execute(self, ctx) -> None:
        task = Task(
            description=self.task_description.value,
            agent=self.agent.value,
            expected_output="Task successfully completed."
        )
        crew = Crew(
            agents=[self.agent.value],
            tasks=[task],
            verbose=True
        )
        self.result.value = crew.kickoff()

@xai_component
class CrewAIConversionTool(Component):
    
    toolbelt_name: InArg[str]

    def execute(self, ctx) -> None:
        class ConversionTools:
            name = "ConversionTools"
            description = "Tool for converting image formats."

            @tool("Convert images")
            def convert_images(description: str):
                """ Extracts conversion parameters from the JSON-formatted description,
                    then converts images in the specified folder to the target format.

                    Expected JSON format:
                    "{\"description\": \"{\\\"input_folder\\\": \\\"screenshots\\\", \\\"output_folder\\\": \\\"my_image\\\", \\\"target_format\\\": \\\"png\\\"}\"}"

                    Returns:
                    - A confirmation message with the output folder.
                """
                try:
                    params = json.loads(description)
                except json.JSONDecodeError:
                    return "Error: Description is not a valid JSON string with conversion parameters."

                input_folder = params.get("input_folder")
                output_folder = params.get("output_folder")
                target_format = params.get("target_format")

                if not input_folder or not output_folder or not target_format:
                    return "Error: Missing one or more required parameters."

                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)

                for filename in os.listdir(input_folder):
                    if filename.lower().endswith(("png", "jpg", "jpeg", "bmp", "gif")):
                        input_path = os.path.join(input_folder, filename)
                        base_name = os.path.splitext(filename)[0]
                        output_path = os.path.join(output_folder, f"{base_name}.{target_format}")
                        try:
                            with Image.open(input_path) as img:
                                img.convert("RGB").save(output_path, target_format.upper())
                        except Exception as e:
                            print(f"Error converting {filename}: {str(e)}")
                return f"All images converted to {target_format.upper()} and saved in {output_folder}"

        toolbelt = self.toolbelt_name.value if self.toolbelt_name.value else "default"

        ctx.setdefault(f"toolbelt_{toolbelt}", {})["ConversionTools.convert_images"] = ConversionTools.convert_images
