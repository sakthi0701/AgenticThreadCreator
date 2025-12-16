from agno.agent import Agent
import sys
import os
import dspy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import dspy_module
from dspy_module import InferTemplate, configure_llm

class TemplateAgent(Agent):
    """
    An agent that infers the correct template for a post
    if one isn't provided.
    """
    
    def __init__(self):
        super().__init__(
            name="TemplateAgent",
            description="Infers a suitable template from a verbose topic.",
        )
        configure_llm()
        self.template_inferrer = dspy.Predict(InferTemplate)
        # This list defines the "options" our LLM can choose from.
        self.available_templates = "['Project Showcase', 'How-to Guide', 'Listicle', 'Lessons Learned', 'Story-Insight-CTA', 'Mistakes to Avoid']"

    def act(self, topic: str, template: str, config: dict = None):
        """
        If the template is valid, pass it through.
        If not, infer a new one.
        """
        # Rotate key before action
        configure_llm()
        
        if config is None:
            config = {}

        # Use available_templates from config if provided, otherwise use default
        current_available_templates = config.get('available_templates', self.available_templates)

        # Check if user provided a real template
        if template and "choose" not in template.lower():
            print(f"✅ TemplateAgent: Using user-provided template: '{template}'")
            return {"final_template": template}
        
        # User asked us to choose, so let's infer one.
        print(f"📝 TemplateAgent: No valid template found. Inferring from topic...")
        result = self.template_inferrer(raw_topic=topic, available_templates=current_available_templates)
        
        # For your hackathon post, this will likely choose "Project Showcase"
        print(f"✅ TemplateAgent: Inferred template: '{result.inferred_template}'")
        return {"final_template": result.inferred_template}