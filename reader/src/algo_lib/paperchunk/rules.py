from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

def normalize_heading_key(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    # Strip leading numbering like "1.", "2.3", "III.", "A.1" (best-effort)
    s = re.sub(r"^\s*([IVXLC]+|\d+)([\.\)]\s*|\s+)", "", s)
    s = re.sub(r"^\s*([A-Z]\.)\s*", "", s)
    return s.lower().strip()

def compile_alias_regex(alias: str) -> re.Pattern:
    """
    Compile alias regex as *whole-phrase containment* within a heading_key.

    This enables:
      - "experiments and results" to match "experiments" and "results"
      - "summary and discussion" to match "summary" and "discussion"
      - numbered headings are already stripped by normalize_heading_key()

    Also supports a small whitelist of safe pluralization on the last token.
    """
    alias = alias.strip().lower()
    tokens = [t for t in re.split(r"\s+", alias) if t]
    if not tokens:
        tokens = [alias]

    plural_whitelist = {
        "work": r"work(?:s)?",
        "experiment": r"experiment(?:s)?",
        "conclusion": r"conclusion(?:s)?",
        "method": r"method(?:s)?",
        "preliminary": r"preliminar(?:y|ies)",
        "result": r"result(?:s)?",
        "limitation": r"limitation(?:s)?",
        "discussion": r"discussion(?:s)?",
        "setting": r"setting(?:s)?",
        "dataset": r"dataset(?:s)?",
        "evaluation": r"evaluation(?:s)?",
    }

    parts: List[str] = []
    for i, tok in enumerate(tokens):
        tok_l = tok.lower()
        if i == len(tokens) - 1 and tok_l in plural_whitelist:
            parts.append(plural_whitelist[tok_l])
        else:
            parts.append(re.escape(tok_l))

    # phrase with flexible whitespace
    phrase = r"\s+".join(parts)

    # "word-ish" boundaries:
    # - avoid matching inside longer tokens
    # - but allow punctuation/whitespace around
    pat = rf"(?:^|[^\w]){phrase}(?:$|[^\w])"
    return re.compile(pat, re.I)

def normalize_alias(s: str) -> str:
    return " ".join(s.lower().strip().split())

@dataclass
class Rules:
    version: int
    compiled_regex_version: int
    selectors: Dict[str, Dict[str, Any]]
    combined_join_tokens: List[str]
    stop_headings: List[str]
    alias_regex: Dict[str, List[Tuple[str, re.Pattern]]]

    @staticmethod
    def load(path: str | Path) -> "Rules":
        p = Path(path)
        yaml_text = p.read_text(encoding="utf-8")
        
        # Check for duplicate selector keys in raw YAML before parsing
        # yaml.safe_load silently overwrites duplicates, so we need to check the AST
        try:
            yaml_doc = yaml.compose(yaml_text)
            selector_keys: List[str] = []
            
            def find_selectors(node):
                """
                Traverse YAML AST to find all selector keys.
                
                Edge cases handled:
                - Standard YAML formats (block/flow style): ✓ Works
                - Multi-line strings: ✓ Works (normalized by parser)
                - Comments: ✓ Ignored by parser
                - Nested structures: ✓ Handled recursively
                
                Edge cases NOT fully handled:
                - YAML anchors/aliases: May not detect duplicates if keys use aliases
                  Example: &method_key method: ... / *method_key: ... (duplicate)
                - Non-scalar keys: If selector key is not a simple string, .value access may fail
                  Example: ? [complex, key]: ... (unlikely in config files)
                - Multi-document YAML: Only first document checked (yaml.compose default)
                  Example: ---\nselectors: ...\n---\nselectors: ... (only first checked)
                - Structure assumptions: Assumes 'selectors:' is direct child of root map
                  If nested differently, duplicates might be missed
                """
                if node.tag == "tag:yaml.org,2002:map":
                    for i in range(0, len(node.value), 2):
                        key_node = node.value[i]
                        value_node = node.value[i + 1] if i + 1 < len(node.value) else None
                        # Edge case: key_node.value assumes scalar node; may fail for complex keys
                        if key_node.value == "selectors" and value_node:
                            # Found selectors section, collect all keys
                            if value_node.tag == "tag:yaml.org,2002:map":
                                for j in range(0, len(value_node.value), 2):
                                    sel_key_node = value_node.value[j]
                                    # Edge case: sel_key_node.value may not resolve correctly for anchors/aliases
                                    selector_keys.append(sel_key_node.value)
                            find_selectors(value_node)
                        elif value_node:
                            find_selectors(value_node)
                elif node.tag == "tag:yaml.org,2002:seq":
                    for item in node.value:
                        find_selectors(item)
            
            if yaml_doc:
                find_selectors(yaml_doc)
            
            # Check for duplicates
            if len(selector_keys) != len(set(selector_keys)):
                duplicates = [key for key in selector_keys if selector_keys.count(key) > 1]
                raise ValueError(
                    f"Duplicate selector IDs found in YAML file {p}: {sorted(set(duplicates))}. "
                    f"Each selector must have a unique name."
                )
        except yaml.YAMLError:
            # If YAML parsing fails, let yaml.safe_load handle it with a proper error
            pass
        
        # Parse YAML (duplicates already checked above)
        obj = yaml.safe_load(yaml_text) or {}
        version = int(obj.get("version", 1))
        crv = int(obj.get("compiled_regex_version", 1))
        selectors = obj.get("selectors", {}) or {}
        join_tokens = obj.get("combined_heading_policy", {}).get("join_tokens", ["and", "&", "/"])
        stop_headings = obj.get("ignore_policy", {}).get("stop_headings", ["references", "bibliography"])

        alias_regex: Dict[str, List[Tuple[str, re.Pattern]]] = {}
        for sel, meta in selectors.items():
            aliases = meta.get("aliases", []) or []
            alias_regex[sel] = [(str(a), compile_alias_regex(str(a))) for a in aliases]

        return Rules(
            version=version,
            compiled_regex_version=crv,
            selectors=selectors,
            combined_join_tokens=list(join_tokens),
            stop_headings=[str(s).lower() for s in stop_headings],
            alias_regex=alias_regex,
        )

    def stop_heading(self, heading_key: str) -> bool:
        return normalize_heading_key(heading_key) in self.stop_headings

    def match_selectors(self, heading_raw: str) -> Tuple[List[str], List[str]]:
        hk = normalize_heading_key(heading_raw)
        matched_sels: List[str] = []
        matched_aliases: List[str] = []
        for sel, pairs in self.alias_regex.items():
            for alias, rgx in pairs:
                if rgx.search(hk):
                    matched_sels.append(sel)
                    matched_aliases.append(alias)
                    break
        return matched_sels, matched_aliases

    def is_combined_heading(self, heading_raw: str) -> bool:
        hk = normalize_heading_key(heading_raw)
        for tok in self.combined_join_tokens:
            if f" {tok} " in f" {hk} ":
                return True
        return False

    def match_combined_parts(self, heading_raw: str) -> Tuple[List[str], List[str]]:
        """
        For combined headings, split by join tokens and match each part separately.
        Returns combined matches from all parts.
        
        Example: "experiments and results" -> matches "experiments" and "results" separately
        """
        hk = normalize_heading_key(heading_raw)
        all_matched_sels: List[str] = []
        all_matched_aliases: List[str] = []
        
        # Find which join token is used
        join_token_used = None
        for tok in self.combined_join_tokens:
            if f" {tok} " in f" {hk} ":
                join_token_used = tok
                break
        
        if not join_token_used:
            return [], []
        
        # Split by the join token
        parts = re.split(rf"\s+{re.escape(join_token_used)}\s+", hk)
        parts = [p.strip() for p in parts if p.strip()]
        
        # Match each part separately
        for part in parts:
            part_sels, part_aliases = self.match_selectors(part)
            all_matched_sels.extend(part_sels)
            all_matched_aliases.extend(part_aliases)
        
        return all_matched_sels, all_matched_aliases
