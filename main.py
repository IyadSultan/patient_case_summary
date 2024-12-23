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
    bodySite: Optional[str] = None
    histology: Optional[str] = None
    stage: Optional[str] = None

    # Additional fields
    stage_grouping: Optional[str] = None
    assessment: Optional[str] = None
    tumor_size: Optional[str] = None
    tumor_marker_results: List[str] = Field(default_factory=list)


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
    given_name: str = Field(default="Not Provided")
    family_name: str = Field(default="Not Provided")
    birth_date: Optional[str] = Field(default="Not Provided")
    age_at_diagnosis: Optional[int] = Field(default=None)
    age: Optional[int] = Field(default=None)
    months_since_diagnosis: Optional[int] = Field(default=None)
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
age: {self.age}
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
You are an assistant that takes a patient's melanoma clinical data and associates the condition with relevant encounters and medications.

**Steps to follow:**
1. Review the patient's demographics, melanoma condition details, recent encounters, and current medications.
2. For the melanoma condition:
   - Determine which encounters are relevant to melanoma diagnosis, treatment, or follow-up
   - Determine which medications are related to melanoma treatment (e.g., immunotherapy, targeted therapy)
   - Consider the stage, body site, and histology of the melanoma when determining relevance
3. Return the final output strictly as a JSON object following the schema.

**Patient Data**:
{patient_info}
"""

GUIDELINE_QUERIES_PROMPT = """\
You are an assistant tasked with determining what melanoma guidelines would be most helpful for this patient's case.

**Instructions:**
1. Review the patient's melanoma details including:
   - Stage of disease
   - Body site affected
   - Histological type
   - Current treatments
2. Consider what clinicians would look up:
   - Staging-specific treatment recommendations
   - Surgical margins based on melanoma type and location
   - Appropriate systemic therapy options
   - Follow-up and monitoring guidelines
3. Formulate 3-5 specific queries focused on melanoma management

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
    """Parse a Synthea-generated (or mCODE-like) patient JSON file into a PatientInfo object."""
    with open(file_path, "r") as f:
        bundle = json.load(f)

    patient_resource = None
    conditions = []
    encounters = []
    medication_requests = []
    observations = []

    # Separate resources by type
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
        elif resource_type == "Observation":
            observations.append(resource)

    # ---------------------
    # 1) Patient Info
    # ---------------------
    if not patient_resource:
        raise ValueError("No Patient resource found in the provided file.")

    name_entry = patient_resource.get("name", [{}])[0]
    # If no 'given' or 'family' is present, handle gracefully
    given_name = name_entry.get("given", ["Not Provided"])[0] if "given" in name_entry else "Not Provided"
    family_name = name_entry.get("family", "Not Provided")
    birth_date = patient_resource.get("birthDate", "Not Provided")
    gender = patient_resource.get("gender", "Not Provided")

    # ---------------------
    # 2) Conditions
    # ---------------------
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

        # Body site
        body_site = None
        if c.get("bodySite"):
            site_info = c["bodySite"][0].get("coding", [{}])[0]
            body_site = site_info.get("display")

        # Stage & Stage Grouping
        stage = None
        stage_grouping = None
        assessment = None

        # mCODE's `stage` array can contain `summary`, `type`, `assessment`
        if "stage" in c and c["stage"]:
            stage_entry = c["stage"][0]  # typically an array
            # stage summary
            stage_summary_coding = stage_entry.get("summary", {}).get("coding", [{}])[0]
            stage = stage_summary_coding.get("display")

            # stage grouping (e.g., T2, T3, etc.)
            type_coding = stage_entry.get("type", {}).get("coding", [{}])[0]
            stage_grouping = type_coding.get("display")

            # assessment (e.g., "pT2aNx")
            assessment_coding = stage_entry.get("assessment", {}).get("coding", [{}])[0]
            assessment = assessment_coding.get("display")

        # Histology (extension)
        histology = None
        for ext in c.get("extension", []):
            if ext.get("url", "").endswith("mcode-histology-morphology-behavior"):
                hist_info = ext.get("valueCodeableConcept", {}).get("coding", [{}])[0]
                histology = hist_info.get("display")

        # Filter out inactive conditions if filter_active is True
        if filter_active:
            if clinical_status == "active":
                cond_obj = ConditionInfo(
                    code=condition_code,
                    display=condition_display,
                    clinical_status=clinical_status,
                    stage=stage,
                    bodySite=body_site,
                    histology=histology,
                    stage_grouping=stage_grouping,
                    assessment=assessment
                )
                condition_info_list.append(cond_obj)
        else:
            cond_obj = ConditionInfo(
                code=condition_code,
                display=condition_display,
                clinical_status=clinical_status,
                stage=stage,
                bodySite=body_site,
                histology=histology,
                stage_grouping=stage_grouping,
                assessment=assessment
            )
            condition_info_list.append(cond_obj)

    # ---------------------
    # 3) Encounters
    # ---------------------
    def get_encounter_date(enc):
        period = enc.get("period", {})
        start = period.get("start")
        if not start:
            return datetime.min
        try:
            if 'T' in start:
                return datetime.strptime(start.split('.')[0], "%Y-%m-%dT%H:%M:%S")
            else:
                return datetime.strptime(start, "%Y-%m-%d")
        except ValueError:
            return datetime.min

    encounters_sorted = sorted(encounters, key=get_encounter_date)
    recent_encounters = encounters_sorted[-3:] if len(encounters_sorted) > 3 else encounters_sorted

    encounter_info_list = []
    for e in recent_encounters:
        period = e.get("period", {})
        start_date = period.get("start", "")
        reason = None
        etype = None
        if "reasonCode" in e and e["reasonCode"]:
            reason = e["reasonCode"][0].get("coding", [{}])[0].get("display")
        if "type" in e and e["type"]:
            etype = e["type"][0].get("coding", [{}])[0].get("display")

        encounter_info_list.append(
            EncounterInfo(
                date=start_date,
                reason_display=reason,
                type_display=etype
            )
        )

    # ---------------------
    # 4) Medications
    # ---------------------
    medication_info_list = []
    for m in medication_requests:
        status = m.get("status")
        if status == "active":
            med_code = m.get("medicationCodeableConcept", {}).get("coding", [{}])[0]
            med_name = med_code.get("display", "Unknown Medication")
            authored = m.get("authoredOn", None)
            if authored:
                authored = authored.split('T')[0]
            dosage_instruction = m.get("dosageInstruction", [{}])[0].get("text", None)

            medication_info_list.append(
                MedicationInfo(
                    name=med_name,
                    start_date=authored,
                    instructions=dosage_instruction
                )
            )

    # ---------------------
    # 5) Observations (Tumor Marker, Size, etc.)
    # ---------------------
    # You can link each observation to a condition if needed, or store them globally
    # in the 'PatientInfo' or create a separate class. Here we just parse them in place.
    for obs in observations:
        code_obj = obs.get("code", {}).get("coding", [{}])[0]
        obs_code = code_obj.get("code")
        obs_display = code_obj.get("display", "")
        obs_value_cc = obs.get("valueCodeableConcept", {}).get("coding", [{}])[0]
        obs_value_display = obs_value_cc.get("display", "")
        
        # Example: if "tumor size" or "tumor marker test" is recognized by LOINC or SNOMED code
        # Collect them to attach to relevant ConditionInfo objects (if known)
        # For demonstration, let's check for "PET/CT" or other tests:
        if "PET/CT" in obs_display:
            # Suppose we want to store this marker in all 'melanoma' conditions
            for cond_obj in condition_info_list:
                if "melanoma" in cond_obj.display.lower():
                    cond_obj.tumor_marker_results.append(obs_value_display)

        # If your Observations have numeric tumor size under `valueQuantity`, you can parse it:
        if "valueQuantity" in obs:
            quantity_val = obs["valueQuantity"].get("value")
            quantity_unit = obs["valueQuantity"].get("unit", "")
            # Attach to relevant condition, or keep a general reference
            for cond_obj in condition_info_list:
                # example check if condition code is for a primary tumor
                if "melanoma" in cond_obj.display.lower():
                    cond_obj.tumor_size = f"{quantity_val} {quantity_unit}"

    # ---------------------
    # 6) Final Assembly
    # ---------------------
    patient_info = PatientInfo(
        given_name=given_name,
        family_name=family_name,
        birth_date=birth_date,
        gender=gender,
        conditions=condition_info_list,
        recent_encounters=encounter_info_list,
        current_medications=medication_info_list,
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
            with open(str(patient_info_path), "r", encoding='utf-8') as f:
                patient_info_dict = json.load(f)
            patient_info = PatientInfo.model_validate(patient_info_dict)
        else:
            if self._verbose:
                ctx.write_event_to_stream(LogEvent(msg=">> Reading patient info"))
            patient_info = parse_synthea_patient(ev.patient_json_path)
            
            if not isinstance(patient_info, PatientInfo):
                raise ValueError(f"Invalid patient info: {patient_info}")
            with open(patient_info_path, "w", encoding='utf-8') as fp:
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
            with open(str(condition_info_path), "r", encoding='utf-8') as f:
                condition_bundles = ConditionBundles.model_validate(
                    json.load(f)
                )
        else:
            condition_bundles = await create_condition_bundles(ev.patient_info)
            with open(condition_info_path, "w", encoding='utf-8') as fp:
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
        # Save match results with UTF-8 encoding
        recs_path = Path(f"{self.output_dir}/guideline_recommendations.jsonl")
        with open(recs_path, "w", encoding='utf-8') as fp:
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
    documents = SimpleDirectoryReader('guidelines').load_data()
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
    handler = workflow.run(patient_json_path="data/melanoma.json")
    
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