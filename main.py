# Patient Case Summary Workflow
# This script processes patient data to generate clinical case summaries with guideline recommendations
# Based on tutorial demonstrating agentic workflow for extracting patient details and checking clinical guidelines

# Standard library imports
import json
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

# Third-party imports
import nest_asyncio
from pydantic import BaseModel, Field

# LlamaIndex imports
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Context,
    Workflow,
    step,
)
from llama_index.core.llms import LLM, ChatMessage, MessageRole
from llama_index.core.prompts import ChatPromptTemplate
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.llms.openai import OpenAI

# Apply nest_asyncio to allow async operations in Jupyter-like environments
nest_asyncio.apply()

# Configure logging
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

#####################################
# Data Model Classes
#####################################

class ConditionInfo(BaseModel):
    """Information about a medical condition."""
    code: str
    display: str
    clinical_status: str

class EncounterInfo(BaseModel):
    """Information about a medical encounter."""
    date: str = Field(..., description="Date of the encounter.")
    reason_display: Optional[str] = Field(None, description="Reason for the encounter.")
    type_display: Optional[str] = Field(None, description="Type or class of the encounter.")

class MedicationInfo(BaseModel):
    """Information about a medication."""
    name: str = Field(..., description="Name of the medication.")
    start_date: Optional[str] = Field(None, description="When the medication was prescribed.")
    instructions: Optional[str] = Field(None, description="Dosage instructions.")

class PatientInfo(BaseModel):
    """Comprehensive patient information including demographics and medical history."""
    given_name: str
    family_name: str
    birth_date: str
    gender: str
    conditions: List[ConditionInfo] = Field(default_factory=list)
    recent_encounters: List[EncounterInfo] = Field(default_factory=list, description="A few recent encounters.")
    current_medications: List[MedicationInfo] = Field(default_factory=list, description="Current active medications.")

    @property
    def demographic_str(self) -> str:
        """Get demographics string."""
        return f"""\
Given name: {self.given_name}
Family name: {self.family_name}
Birth date: {self.birth_date}
Gender: {self.gender}"""

class ConditionBundle(BaseModel):
    """Bundle of condition-specific information including related encounters and medications."""
    condition: ConditionInfo
    encounters: List[EncounterInfo] = Field(default_factory=list)
    medications: List[MedicationInfo] = Field(default_factory=list)

class ConditionBundles(BaseModel):
    """Collection of condition bundles."""
    bundles: List[ConditionBundle]

class GuidelineQueries(BaseModel):
    """Represents queries for retrieving relevant guideline sections."""
    queries: List[str] = Field(
        default_factory=list,
        description="A list of query strings that can be used to search a vector index of medical guidelines."
    )

class GuidelineRecommendation(BaseModel):
    """Recommendation from clinical guidelines."""
    guideline_source: str = Field(..., description="The origin of the guideline (e.g., 'NHLBI Asthma Guidelines').")
    recommendation_summary: str = Field(..., description="A concise summary of the relevant recommendation.")
    reference_section: Optional[str] = Field(None, description="Specific section or reference in the guideline.")

class ConditionSummary(BaseModel):
    """Summary of a condition including recommendations."""
    condition_display: str = Field(..., description="Human-readable name of the condition.")
    summary: str = Field(..., description="A concise narrative summarizing the condition's status.")

class CaseSummary(BaseModel):
    """Complete case summary including all conditions and recommendations."""
    patient_name: str = Field(..., description="The patient's name.")
    age: int = Field(..., description="The patient's age in years.")
    overall_assessment: str = Field(..., description="A high-level summary synthesizing all conditions.")
    condition_summaries: List[ConditionSummary] = Field(
        default_factory=list,
        description="A list of condition-specific summaries."
    )

    def render(self) -> str:
        """Render the case summary as a formatted string."""
        lines = []
        lines.append(f"Patient Name: {self.patient_name}")
        lines.append(f"Age: {self.age} years")
        lines.append("")
        lines.append("Overall Assessment:")
        lines.append(self.overall_assessment)
        lines.append("")
        
        if self.condition_summaries:
            lines.append("Condition Summaries:")
            for csum in self.condition_summaries:
                lines.append(f"- {csum.condition_display}:")
                lines.append(f"  {csum.summary}")
        else:
            lines.append("No specific conditions were summarized.")

        return "\n".join(lines)

#####################################
# Prompt Templates
#####################################

CONDITION_BUNDLE_PROMPT = """\
You are an assistant that takes a patient's summarized clinical data and associates each active condition with any relevant recent encounters and current medications.

**Steps to follow:**
1. Review the patient's demographics, conditions, recent encounters, and current medications.
2. For each condition in 'conditions':
   - Determine which of the 'recent_encounters' are relevant. An encounter is relevant if:
     - The 'reason_display' or 'type_display' of the encounter mentions or is closely related to the condition.
     - Consider synonyms or partial matches. For example, for "Childhood asthma (disorder)", any encounter mentioning "asthma" or "asthma follow-up" is relevant.
   - Determine which of the 'current_medications' are relevant. A medication is relevant if:
     - The medication 'name' or 'instructions' are clearly related to managing that condition. For example, inhalers or corticosteroids for asthma, topical creams for dermatitis.
     - Consider partial matches. For "Atopic dermatitis (disorder)", a medication used for allergic conditions or skin inflammations could be relevant.
3. Ignore patient demographics for relevance determination; they are just context.
4. Return the final output strictly as a JSON object following the schema (provided as a tool call). 
   Do not include extra commentary outside the JSON.

**Patient Data**:
{patient_info}
"""

GUIDELINE_QUERIES_PROMPT = """\
You are an assistant tasked with determining what guidelines would be most helpful to consult for a given patient's condition data. You have:

- Patient information (demographics, conditions, encounters, medications)
- A single condition bundle that includes:
  - One specific condition and its related encounters and medications
- Your goal is to produce several high-quality search queries that can be used to retrieve relevant guideline sections from a vector index of medical guidelines.

**Instructions:**
1. Review the patient info and the condition bundle. Identify the key aspects of the condition that might require guideline consultation—such as disease severity, typical management steps, trigger avoidance, or medication optimization.
2. Consider what clinicians would look up:
   - Best practices for this condition's management
   - Medication recommendations
   - Encounter follow-ups
   - Patient education and preventive measures
3. Formulate 3-5 concise, targeted queries
4. Make the queries condition-specific, incorporating relevant medications or encounter findings. 

Patient Info: {patient_info}

Condition Bundle: {condition_info}
"""

GUIDELINE_RECOMMENDATION_PROMPT = """\
Given the following patient condition and the corresponding relevant medical guideline text (unformatted), 
generate a guideline recommendation according to the schema defined as a tool call.

The condition details are given below. This includes the condition itself, along with associated encounters/medications
that the patient has taken already. Make sure the guideline recommendation is relevant.

**Patient Condition:**
{patient_condition_text}

**Matched Guideline Text(s):**
{guideline_text}
"""

CASE_SUMMARY_SYSTEM_PROMPT = """\
You are a medical assistant that produces a concise and understandable case summary for a clinician. 

You have access to the patient's name, age, and a list of conditions. 

For each condition, you also have related encounters, medications, and guideline recommendations. 

Your goal is to produce a `CaseSummary` object in JSON format that adheres to the CaseSummary schema, defined as a tool call.

**Instructions:**
- Use the patient's name and age as given.
- Create an `overall_assessment` that integrates the data about their conditions, encounters, medications, and guideline recommendations.
- For each condition, write a short `summary` describing:
  - The current state of the condition.
  - Relevant encounters that indicate progress or issues.
  - Medications currently managing that condition and if they align with guidelines.
  - Any key recommendations from the guidelines that should be followed going forward.
- Keep the summaries patient-friendly but medically accurate. Be concise and clear.
- Return only the final JSON that matches the schema. No extra commentary.
"""

CASE_SUMMARY_USER_PROMPT = """\
**Patient Demographics**
{demographic_info}

**Condition Information**
{condition_guideline_info}

Given the above data, produce a `CaseSummary` as per the schema.
"""

#####################################
# Helper Functions
#####################################

def parse_synthea_patient(file_path: str, filter_active: bool = True) -> PatientInfo:
    """Parse a Synthea-generated patient JSON file into a PatientInfo object."""
    # Load the Synthea-generated FHIR Bundle
    with open(file_path, "r") as f:
        bundle = json.load(f)

    patient_resource = None
    conditions = []
    encounters = []
    medication_requests = []

    # Extract resources by type
    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        resource_type = resource.get("resourceType")

        if resource_type == "Patient":
            patient_resource = resource
        elif resource_type == "Condition":
            conditions.append(resource)
        elif resource_type == "Encounter":
            encounters.append(resource)
        elif resource_type == "MedicationRequest":
            medication_requests.append(resource)

    if not patient_resource:
        raise ValueError("No Patient resource found in the provided file.")

    # Extract patient demographics
    name_entry = patient_resource.get("name", [{}])[0]
    given_name = name_entry.get("given", [""])[0]
    family_name = name_entry.get("family", "")
    birth_date = patient_resource.get("birthDate", "")
    gender = patient_resource.get("gender", "")

    # Define excluded conditions
    excluded_conditions = {"Medication review due (situation)", "Risk activity involvement (finding)"}
    
    # Process conditions
    condition_info_list = []
    for c in conditions:
        code_info = c.get("code", {}).get("coding", [{}])[0]
        condition_code = code_info.get("code", "Unknown")
        condition_display = code_info.get("display", "Unknown")
        clinical_status = (
            c.get("clinicalStatus", {})
             .get("coding", [{}])[0]
             .get("code", "unknown")
        )
        
        # Check exclusion and active filters
        if condition_display not in excluded_conditions:
            if filter_active:
                if clinical_status == "active":
                    condition_info_list.append(
                        ConditionInfo(
                            code=condition_code,
                            display=condition_display,
                            clinical_status=clinical_status
                        )
                    )
            else:
                condition_info_list.append(
                    ConditionInfo(
                        code=condition_code,
                        display=condition_display,
                        clinical_status=clinical_status
                    )
                )

    # Parse encounters
    def get_encounter_date(enc):
        period = enc.get("period", {})
        start = period.get("start")
        if not start:
            return datetime.min
        
        # Handle various date formats
        try:
            if 'T' in start:
                clean_date = start.split('.')[0].split('+')[0].split('-')[0]
                return datetime.strptime(clean_date, "%Y-%m-%dT%H:%M:%S")
            else:
                return datetime.strptime(start, "%Y")
        except ValueError:
            return datetime.min

    encounters_sorted = sorted(encounters, key=get_encounter_date)
    recent_encounters = encounters_sorted[-3:] if len(encounters_sorted) > 3 else encounters_sorted

    encounter_info_list = []
    for e in recent_encounters:
        period = e.get("period", {})
        start_date = period.get("start", "")
        if start_date:
            start_date = start_date.split('+')[0].split('-')[0]
        reason = e.get("reasonCode", [{}])[0].get("coding", [{}])[0].get("display", None)
        etype = e.get("type", [{}])[0].get("coding", [{}])[0].get("display", None)
        encounter_info_list.append(
            EncounterInfo(
                date=start_date,
                reason_display=reason,
                type_display=etype
            )
        )

    # Parse medications
    medication_info_list = []
    for m in medication_requests:
        status = m.get("status")
        if status == "active":
            med_code = m.get("medicationCodeableConcept", {}).get("coding", [{}])[0]
            med_name = med_code.get("display", "Unknown Medication")
            authored = m.get("authoredOn", None)
            if authored:
                authored = authored.split('+')[0].split('-')[0]
            dosage_instruction = m.get("dosageInstruction", [{}])[0].get("text", None)
            medication_info_list.append(
                MedicationInfo(
                    name=med_name,
                    start_date=authored,
                    instructions=dosage_instruction
                )
            )

    # Create final PatientInfo object
    patient_info = PatientInfo(
        given_name=given_name,
        family_name=family_name,
        birth_date=birth_date,
        gender=gender,
        conditions=condition_info_list,
        recent_encounters=encounter_info_list,
        current_medications=medication_info_list
    )

    return patient_info

async def create_condition_bundles(
    patient_data: PatientInfo, 
    llm: Optional[LLM] = None
) -> ConditionBundles:
    """Create condition bundles using LLM to associate conditions with encounters/medications."""
    llm = llm or OpenAI(model="gpt-4o")

    prompt = ChatPromptTemplate.from_messages([
        ("user", CONDITION_BUNDLE_PROMPT)
    ])
    condition_bundles = await llm.astructured_predict(
        ConditionBundles,
        prompt,
        patient_info=patient_data.json()
    )

    return condition_bundles

def generate_condition_guideline_str(
    bundle: ConditionBundle,
    rec: GuidelineRecommendation
) -> str:
    """Generate a formatted string combining condition and guideline information."""
    return f"""\
**Condition Info**:
{bundle.json()}

**Recommendation**:
{rec.json()}
"""

#####################################
# Workflow Event Classes
#####################################

class PatientInfoEvent(Event):
    """Event containing patient information."""
    patient_info: PatientInfo

class ConditionBundleEvent(Event):
    """Event containing condition bundles."""
    bundles: ConditionBundles

class MatchGuidelineEvent(Event):
    """Event for matching guidelines to conditions."""
    bundle: ConditionBundle

class MatchGuidelineResultEvent(Event):
    """Event containing results of guideline matching."""
    bundle: ConditionBundle
    rec: GuidelineRecommendation

class GenerateCaseSummaryEvent(Event):
    """Event for generating the final case summary."""
    condition_guideline_info: List[Tuple[ConditionBundle, GuidelineRecommendation]]

class LogEvent(Event):
    """Event for logging messages."""
    msg: str
    delta: bool = False

#####################################
# Main Workflow Class
#####################################

class GuidelineRecommendationWorkflow(Workflow):
    """Workflow for processing patient data and generating guideline recommendations."""

    def __init__(
        self,
        guideline_retriever: BaseRetriever,
        llm: LLM | None = None,
        similarity_top_k: int = 20,
        output_dir: str = "data_out",
        **kwargs,
    ) -> None:
        """Initialize workflow with necessary components."""
        super().__init__(**kwargs)

        self.guideline_retriever = guideline_retriever
        self.llm = llm or OpenAI(model="gpt-4o-mini")
        self.similarity_top_k = similarity_top_k

        # Create output directory if it doesn't exist
        out_path = Path(output_dir) / "workflow_output"
        if not out_path.exists():
            out_path.mkdir(parents=True, exist_ok=True)
            os.chmod(str(out_path), 0o0777)
        self.output_dir = out_path

    @step
    async def parse_patient_info(
        self, ctx: Context, ev: StartEvent
    ) -> PatientInfoEvent:
        """Parse patient information from JSON file."""
        patient_info_path = Path(
            f"{self.output_dir}/patient_info.json"
        )
        if patient_info_path.exists():
            if self._verbose:
                ctx.write_event_to_stream(LogEvent(msg=">> Loading patient info from cache"))
            patient_info_dict = json.load(open(str(patient_info_path), "r"))
            patient_info = PatientInfo.model_validate(patient_info_dict)
        else:
            if self._verbose:
                ctx.write_event_to_stream(LogEvent(msg=">> Reading patient info"))
            patient_info = parse_synthea_patient(ev.patient_json_path)
            
            if not isinstance(patient_info, PatientInfo):
                raise ValueError(f"Invalid patient info: {patient_info}")
            with open(patient_info_path, "w") as fp:
                fp.write(patient_info.model_dump_json())
        if self._verbose:
            ctx.write_event_to_stream(LogEvent(msg=f">> Patient Info: {patient_info.dict()}"))

        await ctx.set("patient_info", patient_info)
        return PatientInfoEvent(patient_info=patient_info)

    @step
    async def create_condition_bundles(
        self, ctx: Context, ev: PatientInfoEvent
    ) -> ConditionBundleEvent:
        """Create bundles of conditions with related encounters and medications."""
        condition_info_path = Path(
            f"{self.output_dir}/condition_bundles.json"
        )
        if condition_info_path.exists():
            condition_bundles = ConditionBundles.model_validate(
                json.load(open(str(condition_info_path), "r"))
            )
        else:
            condition_bundles = await create_condition_bundles(ev.patient_info)
            with open(condition_info_path, "w") as fp:
                fp.write(condition_bundles.model_dump_json())
            
        return ConditionBundleEvent(bundles=condition_bundles)

    @step
    async def dispatch_guideline_match(
        self, ctx: Context, ev: ConditionBundleEvent
    ) -> MatchGuidelineEvent:
        """Dispatch events for matching guidelines to each condition."""
        await ctx.set("num_conditions", len(ev.bundles.bundles))
        
        for bundle in ev.bundles.bundles:
            ctx.send_event(MatchGuidelineEvent(bundle=bundle))

    @step
    async def handle_guideline_match(
        self, ctx: Context, ev: MatchGuidelineEvent
    ) -> MatchGuidelineResultEvent:
        """Handle matching guidelines to a specific condition."""
        patient_info = await ctx.get("patient_info")
        
        # Generate queries to find relevant guidelines
        prompt = ChatPromptTemplate.from_messages([
            ("user", GUIDELINE_QUERIES_PROMPT)
        ])
        guideline_queries = await self.llm.astructured_predict(
            GuidelineQueries,
            prompt,
            patient_info=patient_info.demographic_str,
            condition_info=ev.bundle.json()
        )

        # Retrieve and process matching guidelines
        guideline_docs_dict = {}
        for query in guideline_queries.queries:
            if self._verbose:
                ctx.write_event_to_stream(LogEvent(msg=f">> Generating query: {query}"))
            cur_guideline_docs = self.guideline_retriever.retrieve(query)
            guideline_docs_dict.update({
                d.id_: d for d in cur_guideline_docs
            })
        guideline_docs = guideline_docs_dict.values()
        guideline_text="\n\n".join([g.get_content() for g in guideline_docs])
        if self._verbose:
            ctx.write_event_to_stream(
                LogEvent(msg=f">> Found guidelines: {guideline_text[:200]}...")
            )
        
        # Generate guideline recommendation
        prompt = ChatPromptTemplate.from_messages([
            ("user", GUIDELINE_RECOMMENDATION_PROMPT)
        ])
        guideline_rec = await self.llm.astructured_predict(
            GuidelineRecommendation,
            prompt,
            patient_condition_text=ev.bundle.json(),
            guideline_text=guideline_text
        )
        if self._verbose:
            ctx.write_event_to_stream(
                LogEvent(msg=f">> Guideline recommendation: {guideline_rec.json()}")
            )
        
        if not isinstance(guideline_rec, GuidelineRecommendation):
            raise ValueError(f"Invalid guideline recommendation: {guideline_rec}")

        return MatchGuidelineResultEvent(bundle=ev.bundle, rec=guideline_rec)

    @step
    async def gather_guideline_match(
        self, ctx: Context, ev: MatchGuidelineResultEvent
    ) -> GenerateCaseSummaryEvent:
        """Gather all guideline matching results."""
        num_conditions = await ctx.get("num_conditions")
        events = ctx.collect_events(ev, [MatchGuidelineResultEvent] * num_conditions)
        if events is None:
            return

        match_results = [(e.bundle, e.rec) for e in events]
        # Save match results
        recs_path = Path(f"{self.output_dir}/guideline_recommendations.jsonl")
        with open(recs_path, "w") as fp:
            for _, rec in match_results:
                fp.write(rec.model_dump_json() + "\n")
                
        return GenerateCaseSummaryEvent(condition_guideline_info=match_results)

    @step
    async def generate_output(
        self, ctx: Context, ev: GenerateCaseSummaryEvent
    ) -> StopEvent:
        """Generate final case summary."""
        if self._verbose:
            ctx.write_event_to_stream(LogEvent(msg=">> Generating Case Summary"))

        patient_info = await ctx.get("patient_info")
        demographic_info = patient_info.demographic_str

        condition_guideline_strs = []
        for condition_bundle, guideline_rec in ev.condition_guideline_info:
            condition_guideline_strs.append(
                generate_condition_guideline_str(condition_bundle, guideline_rec)
            )
        condition_guideline_str = "\n\n".join(condition_guideline_strs)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", CASE_SUMMARY_SYSTEM_PROMPT),
            ("user", CASE_SUMMARY_USER_PROMPT)
        ])
        case_summary = await self.llm.astructured_predict(
            CaseSummary,
            prompt,
            demographic_info=demographic_info,
            condition_guideline_info=condition_guideline_str
        )

        return StopEvent(result={"case_summary": case_summary})

#####################################
# Main Execution
#####################################

async def main():
    # Initialize workflow components
    documents = SimpleDirectoryReader('guidelines/').load_data()
    local_index = VectorStoreIndex.from_documents(documents)
    retriever = local_index.as_retriever(similarity_top_k=3)

    llm = OpenAI(model="gpt-4o")
    workflow = GuidelineRecommendationWorkflow(
        guideline_retriever=retriever,
        llm=llm,
        verbose=True,
        timeout=None
    )

    # Run workflow
    handler = workflow.run(patient_json_path="data/almeta_buckridge.json")
    
    # Process events
    async for event in handler.stream_events():
        if isinstance(event, LogEvent):
            if event.delta:
                print(event.msg, end="")
            else:
                print(event.msg)

    # Get and display results
    response_dict = await handler
    case_summary = response_dict["case_summary"]
    print(case_summary.render())

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())