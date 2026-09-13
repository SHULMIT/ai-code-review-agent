from review_agent.graph.state import ReviewState
from review_agent.models.finding import Component
from review_agent.tools.component_classifier import ComponentMapping, classify_file


# LangGraph nodes must accept only `state` - external dependencies are bound here via closure before the node is registered.
def make_classify_node(component_mapping: ComponentMapping):
    """Build the classify node with the already-parsed component mapping bound via closure.

    Takes the parsed mapping, not a path. build_graph reads and validates
    config/component_mapping.yaml once and hands the result to every node
    that needs it, so no node reaches for the filesystem while a review is
    running - the same rule the reviewer nodes already follow for their
    prompt templates.

    This node stays the one place a mapping failure is reported. The
    reviewers receive the same (empty) rules and classify everything as
    Component.UNKNOWN, but saying so six more times would repeat one root
    cause under six prefixes rather than add information.
    """

    def classify_node(state: ReviewState) -> dict:
        """Classify every changed file into a DataNet component and dedupe the result.

        A missing file, invalid YAML, or an unknown component name was
        already turned into component_mapping.load_error at graph-build
        time; it is recorded into the state's errors field here rather than
        crashing the node.
        """
        if not state.changed_files:
            return {"components": [], "errors": ["classify_node: no changed_files in state"]}

        if component_mapping.load_error is not None:
            return {
                "components": [],
                "errors": [f"classify_node: {component_mapping.load_error}"],
            }

        components: list[Component] = []
        for file_path in state.changed_files:
            component = classify_file(file_path, component_mapping.rules)
            if component not in components:
                components.append(component)

        return {"components": components}

    return classify_node
