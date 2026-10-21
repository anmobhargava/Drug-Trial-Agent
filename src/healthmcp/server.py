"""
The MCP server. It exposes drug and clinical-trial lookups as tools that any MCP client (Claude Desktop, an agent, the inspector page in this repo) can discover and call.

What a client sees over the protocol:
  tools     5 read-only lookups, each with a typed input schema and a plain description the model reads to decide when to use it
  resource  health://about: what the data is, and what this server deliberately does NOT provide
  prompt    drug_safety_brief: a reusable starting message that tells a model which tools to call, in what order

Design choices worth knowing:
  * Every tool is marked read-only. A client can use that hint to expose only safe tools (Agent 2 does).
  * Dosing is left out on purpose. Label text for dosing is easy to misread and this server is for looking things up, not for advice.
  * Results are trimmed (label sections are huge) and always carry a source link, so whoever reads them can check.
  * Errors come back as tool errors with a message the model can act on ("no matches for X", "source unavailable, try again"), not as crashes.
"""

import re
from typing import Annotated, Literal

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from .api import CTGOV, OPENFDA, NoMatch, SourceError

SECTIONS = {"indications": "indications_and_usage", "warnings": "warnings", "contraindications": "contraindications",
            "interactions": "drug_interactions", "adverse_reactions": "adverse_reactions"}
MAX_TEXT = 2000
READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True)

ABOUT = """Healthcare lookup server.

Data: drug labels and adverse-event report counts from openFDA, clinical trials from ClinicalTrials.gov (or fictional demo data when started with --demo-data).

What it is for: looking things up and reading what the sources say.
What it is NOT for: dosing, diagnosis, or deciding what anyone should take. Dosing sections are deliberately not exposed.
Adverse-event counts are spontaneous reports. They do not show that a drug caused a reaction and they are not rates."""


def clip(text: str, n: int = MAX_TEXT) -> tuple[str, bool]:
    text = re.sub(r"\s+", " ", text or "").strip()
    return (text[: n - 1].rstrip() + "…", True) if len(text) > n else (text, False)


def _q(name: str) -> str:
    return name.replace('"', "").strip()


async def _first_label(api, name: str) -> dict:
    n = _q(name)
    try:
        data = await api.get(f"{OPENFDA}/drug/label.json", {"search": f'openfda.brand_name:"{n}" OR openfda.generic_name:"{n}"', "limit": 1})
    except NoMatch:
        raise ToolError(f"No drug label found for '{name}'. Check the spelling, or try the generic name.")
    except SourceError as e:
        raise ToolError(str(e))
    return data["results"][0]


def _names(label: dict) -> dict:
    o = label.get("openfda", {})
    return {"brand_names": o.get("brand_name", []), "generic_names": o.get("generic_name", []), "manufacturer": (o.get("manufacturer_name") or [""])[0]}


def _dailymed(set_id: str) -> str:
    return f"https://dailymed.nlm.nih.gov/dailymed/lookup.cfm?setid={set_id}" if not set_id.startswith("demo-") else f"demo:{set_id}"


def build_server(api) -> FastMCP:
    mcp = FastMCP("healthmcp", instructions="Look up drug labels, adverse-event report counts and clinical trials. Read-only. Not for dosing or medical advice.", log_level="WARNING")

    @mcp.tool(annotations=READ_ONLY.model_copy(update={"title": "Search drug labels"}))
    async def search_drug_labels(name: Annotated[str, Field(min_length=2, max_length=80, description="Brand or generic drug name, e.g. 'ibuprofen'")],
                                 limit: Annotated[int, Field(ge=1, le=10, description="How many labels to return")] = 5) -> dict:
        """Find drug labels by brand or generic name. Returns each label's id, names and manufacturer. Use this first when you are not sure a drug exists or what it is called."""
        n = _q(name)
        try:
            data = await api.get(f"{OPENFDA}/drug/label.json", {"search": f'openfda.brand_name:"{n}" OR openfda.generic_name:"{n}"', "limit": limit})
        except NoMatch:
            raise ToolError(f"No drug label found for '{name}'. Check the spelling, or try the generic name.")
        except SourceError as e:
            raise ToolError(str(e))
        drugs = [{"set_id": r.get("set_id", ""), **_names(r)} for r in data["results"]]
        return {"query": name, "total": data.get("meta", {}).get("results", {}).get("total", len(drugs)), "drugs": drugs, "source": api.name}

    @mcp.tool(annotations=READ_ONLY.model_copy(update={"title": "Read a label section"}))
    async def get_label_section(name: Annotated[str, Field(min_length=2, max_length=80, description="Brand or generic drug name")],
                                section: Annotated[Literal["indications", "warnings", "contraindications", "interactions", "adverse_reactions"],
                                                   Field(description="Which part of the label to read. There is no dosing option on purpose.")]) -> dict:
        """Read one section of a drug's official label: what it is for, warnings, contraindications, interactions with other drugs, or adverse reactions. Text is trimmed and carries a source link."""
        label = await _first_label(api, name)
        raw = label.get(SECTIONS[section])
        if not raw:
            raise ToolError(f"The label for '{name}' has no '{section}' section in this data.")
        text, truncated = clip(" ".join(raw))
        return {**_names(label), "section": section, "text": text, "truncated": truncated, "source_url": _dailymed(label.get("set_id", "")), "source": api.name}

    @mcp.tool(annotations=READ_ONLY.model_copy(update={"title": "Adverse event report counts"}))
    async def adverse_event_summary(name: Annotated[str, Field(min_length=2, max_length=80, description="Brand or generic drug name")],
                                    top: Annotated[int, Field(ge=1, le=20, description="How many of the most-reported reactions")] = 10) -> dict:
        """The reactions most often named in spontaneous adverse-event reports for a drug, with report counts. Reports are not proof a drug caused a reaction, and the counts are not rates."""
        n = _q(name)
        try:
            data = await api.get(f"{OPENFDA}/drug/event.json", {"search": f'patient.drug.medicinalproduct:"{n}"', "count": "patient.reaction.reactionmeddrapt.exact", "limit": top})
        except NoMatch:
            raise ToolError(f"No adverse-event reports found for '{name}'.")
        except SourceError as e:
            raise ToolError(str(e))
        return {"drug": name, "top_reactions": [{"reaction": r["term"], "reports": r["count"]} for r in data["results"]],
                "caveat": "Spontaneous reports. They do not show the drug caused the reaction and are not rates.", "source": api.name}

    @mcp.tool(annotations=READ_ONLY.model_copy(update={"title": "Search clinical trials"}))
    async def search_trials(condition: Annotated[str, Field(min_length=3, max_length=100, description="A condition or disease, e.g. 'type 2 diabetes'")],
                            status: Annotated[Literal["recruiting", "completed", "any"], Field(description="Trial status filter")] = "recruiting",
                            limit: Annotated[int, Field(ge=1, le=10)] = 5) -> dict:
        """Search ClinicalTrials.gov for studies about a condition. Returns each trial's NCT id, title, status, phase and sponsor. Use get_trial_details for one trial's eligibility."""
        params = {"query.cond": condition, "pageSize": limit, "countTotal": "true"}
        if status != "any":
            params["filter.overallStatus"] = status.upper()
        try:
            data = await api.get(f"{CTGOV}/studies", params)
        except SourceError as e:
            raise ToolError(str(e))
        trials = []
        for s in data.get("studies", []):
            p = s["protocolSection"]
            trials.append({"nct_id": p["identificationModule"]["nctId"], "title": p["identificationModule"].get("briefTitle", ""), "status": p["statusModule"].get("overallStatus", ""),
                           "phases": p.get("designModule", {}).get("phases", []), "conditions": p.get("conditionsModule", {}).get("conditions", []),
                           "sponsor": p.get("sponsorCollaboratorsModule", {}).get("leadSponsor", {}).get("name", "")})
        return {"condition": condition, "total": data.get("totalCount", len(trials)), "trials": trials, "source": api.name}

    @mcp.tool(annotations=READ_ONLY.model_copy(update={"title": "Trial details"}))
    async def get_trial_details(nct_id: Annotated[str, Field(pattern=r"^NCT\d{8}$", description="A ClinicalTrials.gov id like NCT01234567")]) -> dict:
        """Details for one trial: summary, phase, sponsor, number of sites and the eligibility criteria (trimmed). Needs an NCT id from search_trials."""
        try:
            data = await api.get(f"{CTGOV}/studies/{nct_id}")
        except NoMatch:
            raise ToolError(f"No trial found with id {nct_id}.")
        except SourceError as e:
            raise ToolError(str(e))
        p = data["protocolSection"]
        elig, trunc = clip(p.get("eligibilityModule", {}).get("eligibilityCriteria", ""))
        return {"nct_id": nct_id, "title": p["identificationModule"].get("briefTitle", ""), "status": p["statusModule"].get("overallStatus", ""), "phases": p.get("designModule", {}).get("phases", []),
                "sponsor": p.get("sponsorCollaboratorsModule", {}).get("leadSponsor", {}).get("name", ""), "summary": clip(p.get("descriptionModule", {}).get("briefSummary", ""), 800)[0],
                "eligibility": elig, "eligibility_truncated": trunc, "ages": [p.get("eligibilityModule", {}).get(k) for k in ("minimumAge", "maximumAge")],
                "sites": len(p.get("contactsLocationsModule", {}).get("locations", [])), "source_url": f"https://clinicaltrials.gov/study/{nct_id}" if not nct_id.startswith("NCT9") else f"demo:{nct_id}",
                "source": api.name}

    @mcp.resource("health://about", mime_type="text/plain")
    def about() -> str:
        """What this server's data is, and what it is deliberately not for."""
        return ABOUT + f"\nData source now: {api.name}"

    @mcp.prompt()
    def drug_safety_brief(drug: str) -> str:
        """A starting message that asks a model to brief someone on a drug using these tools."""
        return (f"Give a short, sourced safety brief on {drug}. Call search_drug_labels first to confirm the drug, then get_label_section for 'warnings' and 'interactions', "
                f"then adverse_event_summary. Quote only what the tools return, include the source links, and note that adverse-event counts are not rates. Do not give dosing or personal medical advice.")

    return mcp
