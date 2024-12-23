# Manual

This document explains **how** the Patient Case Summary application processes patient data step by step, describing the internal flow of its functions and how they work together to generate a final clinical summary with guideline recommendations. It is not a step-by-step usage manual, but rather a look “under the hood” at how the script is structured.

---

## Overview of What the Script Does

The script processes a FHIR-like JSON file (e.g., from [Synthea](https://github.com/synthetichealth/synthea)) that contains patient demographics, conditions, encounters, and medication requests. Then it:

1. **Parses** the JSON into a structured `PatientInfo` object.
2. **Bundles** each condition with relevant encounters and medications.
3. **Queries** clinical guidelines for each condition using a Large Language Model (LLM).
4. **Generates** guideline recommendations.
5. **Compiles** all this information into a comprehensive `CaseSummary` object (in JSON format).

The result is a concise summary of the patient’s case, including relevant guidelines and recommendations for managing each condition.

---

## Key Components

### 1. Data Model Classes

These classes (e.g., `PatientInfo`, `ConditionInfo`, `EncounterInfo`, etc.) represent the building blocks of the patient’s data:

- **`PatientInfo`**  
  Holds demographics, conditions, encounters, and current medications.

- **`ConditionInfo`**  
  Information about a single condition, including code, display, clinical status, stage, body site, and histology.

- **`EncounterInfo`**  
  Represents an encounter event, storing the date, reason, and type.

- **`MedicationInfo`**  
  Information about a prescribed medication, including the name, start date, and instructions.

- **`ConditionBundle`**  
  Once the app associates a `ConditionInfo` with specific encounters and medications (as relevant to that condition), it stores these items in a `ConditionBundle`.

- **`GuidelineQueries`, `GuidelineRecommendation`, `ConditionSummary`, `CaseSummary`**  
  These classes store the queries that an LLM will generate for guidelines, the actual recommendation text retrieved and summarized, and the final condition or case summaries.

---

### 2. Prompt Templates

These prompts define how the LLM should be instructed at various steps:

- **`CONDITION_BUNDLE_PROMPT`**  
  Instructs the LLM to group encounters/medications based on relevance to a given condition.

- **`GUIDELINE_QUERIES_PROMPT`**  
  Directs the LLM to produce queries for retrieving relevant guidelines.

- **`GUIDELINE_RECOMMENDATION_PROMPT`**  
  Tells the LLM how to output a `GuidelineRecommendation` based on retrieved guideline text.

- **`CASE_SUMMARY_SYSTEM_PROMPT`** & **`CASE_SUMMARY_USER_PROMPT`**  
  Combined to instruct the LLM on how to produce the final `CaseSummary` JSON.

These prompts are crucial because they specify **what** the LLM should do with the data at each stage of the workflow.

---

### 3. Helper Functions

- **`parse_synthea_patient(file_path: str, filter_active: bool = True) -> PatientInfo`**  
  Parses the JSON file from Synthea/FHIR into a `PatientInfo` object.  
  1. It reads the bundle.  
  2. Extracts the Patient resource, Condition resources, Encounter resources, and MedicationRequest resources.  
  3. Filters out inactive or irrelevant conditions/medications if `filter_active` is True.  
  4. Returns a `PatientInfo` instance encapsulating all relevant data.

- **`create_condition_bundles(patient_data: PatientInfo, llm: Optional[LLM]) -> ConditionBundles`**  
  Uses an LLM to figure out which medications and encounters relate to which conditions. It:
  1. Feeds patient demographics, conditions, encounters, and medications to the LLM via a structured prompt.  
  2. The LLM returns JSON describing each condition alongside the relevant encounters and meds.  
  3. This data is stored in a `ConditionBundles` object.

- **`generate_condition_guideline_str(bundle: ConditionBundle, rec: GuidelineRecommendation) -> str`**  
  Produces a formatted text snippet that combines condition information with the final guideline recommendation for easy reference or debugging.

---

### 4. Workflow Classes and Events

The application uses a workflow approach with “events” flowing from step to step:

1. **Events**  
   - `PatientInfoEvent` – Contains the parsed `PatientInfo`.  
   - `ConditionBundleEvent` – Contains the `ConditionBundles`.  
   - `MatchGuidelineEvent` – Signals that we need to find relevant guidelines for a single condition.  
   - `MatchGuidelineResultEvent` – The result of guideline retrieval and summarization for one condition.  
   - `GenerateCaseSummaryEvent` – Signals we have all condition–guideline pairs and can create the final summary.  
   - `LogEvent` – Carries log messages for debugging or progress updates.

2. **Workflow**: `GuidelineRecommendationWorkflow`  
   - **`parse_patient_info`** (step)  
     * Reads or caches patient data from JSON.  
     * Emits a `PatientInfoEvent` that holds a `PatientInfo`.
     
   - **`create_condition_bundles`** (step)  
     * Calls `create_condition_bundles(...)` to get a `ConditionBundles` object.  
     * Emits a `ConditionBundleEvent`.
     
   - **`dispatch_guideline_match`** (step)  
     * For each condition in `ConditionBundles`, emits a `MatchGuidelineEvent`.
     
   - **`handle_guideline_match`** (step)  
     * For each condition (in `MatchGuidelineEvent`), it:  
       1. Uses `GUIDELINE_QUERIES_PROMPT` to generate queries for guidelines.  
       2. Retrieves guidelines from a vector store index.  
       3. Uses `GUIDELINE_RECOMMENDATION_PROMPT` to produce a `GuidelineRecommendation`.  
     * Emits a `MatchGuidelineResultEvent`.
     
   - **`gather_guideline_match`** (step)  
     * Waits until all `MatchGuidelineResultEvent` events are collected (i.e., for each condition).  
     * Bundles them into a single `GenerateCaseSummaryEvent`.
     
   - **`generate_output`** (step)  
     * Gathers patient data and all condition–guideline pairs.  
     * Uses the `CASE_SUMMARY_SYSTEM_PROMPT` + `CASE_SUMMARY_USER_PROMPT` to produce a final `CaseSummary`.  
     * Emits a `StopEvent` with the summary.

Essentially, each step in the workflow corresponds to an “event handler” that waits for a specific input event, performs an action (often via an LLM prompt or some data-processing function), and then produces another event to propagate through the flow.

---

### 5. Putting It All Together: How the Final Decision Is Reached

1. **Load & Parse Patient Data**  
   The script reads a JSON file (`parse_patient_info`). This transforms the raw data (bundles of conditions, encounters, etc.) into a structured `PatientInfo`.

2. **Associate Conditions with Encounters/Medications**  
   Next, it calls `create_condition_bundles`. The LLM is given a summary of the patient data and must decide which encounters/medications relate to each condition.

3. **Generate Guideline Queries**  
   For each `ConditionBundle`, the LLM (via `GUIDELINE_QUERIES_PROMPT`) invents 3–5 queries to search an internal guideline library.

4. **Retrieve Guidelines**  
   Those queries are run against a vector store index of guideline documents. The relevant text passages are concatenated.

5. **Summarize Recommendations**  
   With `GUIDELINE_RECOMMENDATION_PROMPT`, the LLM processes both (a) the condition data and (b) the retrieved guideline text, producing a structured `GuidelineRecommendation` for that condition.

6. **Compile Final Case Summary**  
   Once all conditions have a `GuidelineRecommendation`, the script packages them (via `CASE_SUMMARY_SYSTEM_PROMPT` and `CASE_SUMMARY_USER_PROMPT`) into a single `CaseSummary`. This final summary includes:
   - The patient’s name and age.  
   - An overall assessment of the conditions, encounters, and medications.  
   - Condition-by-condition summaries and relevant recommendations.

7. **Output**  
   The script outputs a fully structured `CaseSummary` object, which can be rendered as a human-readable report (the `.render()` method) or used in other systems as JSON.

---

## Conclusion

The Patient Case Summary app uses a multi-step workflow, combining traditional Python parsing and data manipulation with LLM-powered text generation. Each step in the workflow handles a specific task:  
1) reading patient data,  
2) bundling conditions,  
3) retrieving guidelines,  
4) generating recommendations,  
5) summarizing everything into a clean final report.

This separation of tasks allows for clarity, modularity, and easy debugging. By the end, clinicians or researchers have a concise but comprehensive summary of a patient’s case and the latest guideline recommendations for that patient’s conditions.