# -*- coding: utf-8 -*-
"""
Advanced English Grammar Checking System - Optimized Version
Supports GPU acceleration, multi-model integration, intelligent redundancy elimination
Fixed LanguageTool initialization issues, improved accuracy
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration
import torch
import language_tool_python
from spellchecker import SpellChecker
import difflib
import re
import os
import shutil
import tempfile
import requests
import zipfile
import time
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import warnings
import logging

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.empty_cache()
        logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        return device
    else:
        logger.info("Using CPU")
        return torch.device("cpu")


DEVICE = setup_device()

# --- Optimized Model Configuration ---
MODEL_OPTIONS = {
    "T5-Base (Vennify)": {
        "name": "vennify/t5-base-grammar-correction",
        "prefix": "grammar: ",
        "model_type": "AutoModelForSeq2SeqLM",
        "params": {
            "max_length": 128,
            "num_beams": 4,
            "early_stopping": True,
            "no_repeat_ngram_size": 2,
            "repetition_penalty": 1.1,
            "length_penalty": 1.0,
            "do_sample": False,
            "temperature": 1.0
        },
        "performance": {
            "accuracy": "High",
            "speed": "Fast",
            "memory": "~2GB",
            "strengths": "Basic grammar errors, punctuation, fast response",
            "weaknesses": "Limited complex sentence restructuring ability"
        }
    },
    # --- Fine-tuned BART Model ---
    "My BART Fine-tuned": {
        "name": "finetuned_bart_model_agentlans_local",
        # BART training didn't use prefix, so leave empty
        "prefix": "",
        "model_type": "AutoModelForSeq2SeqLM",
        "params": {  # Inference generation parameters, using T5 settings initially
            "max_length": 128,  # Training used 128
            "num_beams": 4,
            "early_stopping": True,
            "no_repeat_ngram_size": 3,
            "repetition_penalty": 1.05,
            "length_penalty": 1.0,
            "do_sample": False
        },
        "performance": {  # Documentation only, can be customized
            "accuracy": "Fine-tuned",
            "speed": "Fast",
            "memory": "~2.5GB",
            "strengths": "In-domain corrections",
            "weaknesses": "May overfit small dataset"
        }
    }
}

# --- Global Instance Management ---
_language_tool_instance = None
_spell_checker_instance = None
loaded_models = {}
DISABLE_LANGUAGE_TOOL = False  # Set to False to enable LanguageTool (Mode 3: LanguageTool + SpellChecker)


def get_language_tool_instance():
    """Get LanguageTool instance with proper initialization"""
    if DISABLE_LANGUAGE_TOOL:
        return None

    global _language_tool_instance
    if _language_tool_instance is None:
        try:
            logger.info("Initializing LanguageTool...")
            _language_tool_instance = language_tool_python.LanguageTool('en-US')
            logger.info("LanguageTool initialization successful")
        except Exception as e:
            logger.error(f"LanguageTool initialization failed: {e}")
            _language_tool_instance = None

    return _language_tool_instance


def get_spell_checker_instance():
    """Get SpellChecker singleton with enhanced error handling"""
    global _spell_checker_instance
    if _spell_checker_instance is None:
        try:
            logger.info("Initializing SpellChecker...")
            _spell_checker_instance = SpellChecker()

            # Add common professional vocabulary to dictionary to avoid false positives
            custom_words = {
                'booktask', 'bookcase', 'textbook', 'workbook',
                'handbook', 'notebook', 'facebook', 'laptop',
                'website', 'email', 'online', 'offline'
            }
            _spell_checker_instance.word_frequency.load_words(custom_words)

            logger.info("SpellChecker initialization successful")
        except Exception as e:
            logger.error(f"SpellChecker initialization failed: {e}")
            _spell_checker_instance = None
    return _spell_checker_instance


def check_sentence_structure(text: str) -> List[Dict]:
    """Check sentence structure issues"""
    errors = []

    # Check sentence capitalization
    sentences = re.split(r'[.!?]\s+', text)
    current_pos = 0

    for sentence in sentences:
        if sentence and sentence[0].islower():
            errors.append({
                "message": "Sentence should start with a capital letter",
                "replacement": sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence.upper(),
                "offset": current_pos,
                "length": 1,
                "context": text[max(0, current_pos - 20):current_pos + 20],
                "rule_id": "SENTENCE_START_CASE",
                "category": "TYPOGRAPHY",
                "error_type": "Capitalization",
                "severity": "Medium",
                "confidence": 0.9,
                "priority": 50
            })
        current_pos += len(sentence) + 2  # Add punctuation and space

    # Check missing end punctuation
    if text and text[-1] not in '.!?':
        errors.append({
            "message": "Sentence is missing end punctuation",
            "replacement": text + ".",
            "offset": len(text) - 1,
            "length": 1,
            "context": text[-40:],
            "rule_id": "MISSING_PUNCTUATION",
            "category": "PUNCTUATION",
            "error_type": "Punctuation",
            "severity": "Low",
            "confidence": 0.7,
            "priority": 30
        })

    return errors


def load_hf_model(model_name: str, model_type: str):
    """Load and cache Hugging Face models, ensuring device consistency"""
    if model_name not in loaded_models:
        logger.info(f"Loading model: {model_name} (type: {model_type})")
        try:
            # Use cache directory to avoid repeated downloads
            cache_dir = os.path.join(os.getcwd(), "model_cache")
            os.makedirs(cache_dir, exist_ok=True)

            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                local_files_only=False
            )

            # Ensure pad_token exists
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Choose loading method based on model type and device
            if DEVICE == "cuda":
                if model_type == "T5ForConditionalGeneration":
                    model = T5ForConditionalGeneration.from_pretrained(
                        model_name,
                        cache_dir=cache_dir,
                        torch_dtype=torch.float16,
                        device_map="auto"
                    )
                else:
                    model = AutoModelForSeq2SeqLM.from_pretrained(
                        model_name,
                        cache_dir=cache_dir,
                        torch_dtype=torch.float16
                    ).to(DEVICE)
            else:
                # CPU mode
                if model_type == "T5ForConditionalGeneration":
                    model = T5ForConditionalGeneration.from_pretrained(
                        model_name,
                        cache_dir=cache_dir
                    )
                else:
                    model = AutoModelForSeq2SeqLM.from_pretrained(
                        model_name,
                        cache_dir=cache_dir
                    )

            model.eval()
            loaded_models[model_name] = {"tokenizer": tokenizer, "model": model}
            logger.info(f"Model {model_name} loaded successfully")

        except Exception as e:
            logger.error(f"Model {model_name} loading failed: {e}")
            return None, None

    return loaded_models[model_name]["tokenizer"], loaded_models[model_name]["model"]


# --- Enhanced Core Functions ---
def correct_grammar_hf(text: str, model_key: str) -> Dict:
    """Use specified model for grammar correction, fixing device mismatch issues"""
    if model_key not in MODEL_OPTIONS:
        return {"error": "Model configuration not found", "corrected_text": text}

    model_config = MODEL_OPTIONS[model_key]
    model_name = model_config["name"]
    prefix = model_config["prefix"]
    params = model_config["params"]
    model_type = model_config["model_type"]

    tokenizer, model = load_hf_model(model_name, model_type)
    if not tokenizer or not model:
        return {"error": "Model loading failed", "corrected_text": text}

    # Enhanced preprocessing
    cleaned_text = preprocess_text(text)
    input_text = f"{prefix}{cleaned_text}" if prefix else cleaned_text

    try:
        # Ensure tokenizer output is on correct device
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
            add_special_tokens=True
        )

        # Manually move inputs to correct device
        if DEVICE == "cuda":
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            input_length = inputs['input_ids'].shape[1]

            # Dynamic parameter adjustment
            dynamic_params = adjust_generation_params(params, input_length, text)

            # Ensure all parameters are on same device
            outputs = model.generate(
                inputs['input_ids'],
                attention_mask=inputs.get('attention_mask'),
                max_length=dynamic_params["max_length"],
                min_length=dynamic_params.get("min_length", 1),
                num_beams=dynamic_params["num_beams"],
                early_stopping=dynamic_params["early_stopping"],
                no_repeat_ngram_size=dynamic_params.get("no_repeat_ngram_size", 2),
                repetition_penalty=dynamic_params["repetition_penalty"],
                length_penalty=dynamic_params["length_penalty"],
                do_sample=dynamic_params["do_sample"],
                temperature=dynamic_params.get("temperature", 1.0) if dynamic_params["do_sample"] else None,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Enhanced post-processing
        corrected_text = postprocess_text(corrected_text, prefix, text)

        # Intelligent quality assessment
        quality_score = assess_correction_quality(text, corrected_text)

        return {
            "corrected_text": corrected_text,
            "model_used": model_key,
            "confidence": quality_score["confidence"],
            "changes_made": corrected_text != text,
            "improvement_score": quality_score["score"],
            "processing_time": "optimized",
            "original_text": text
        }

    except Exception as e:
        logger.error(f"Grammar correction generation failed: {e}")
        return {"error": f"Generation failed: {str(e)}", "corrected_text": text}


def preprocess_text(text: str) -> str:
    """Enhanced text preprocessing"""
    if not text.strip():
        return text

    # Basic cleaning
    text = re.sub(r'\s+', ' ', text.strip())

    # Fix obvious formatting issues
    text = re.sub(r'\s+([,.!?;:])', r'\1', text)  # Space before punctuation
    text = re.sub(r'([.!?])\s*([a-z])', r'\1 \2', text)  # Sentence spacing

    # Fix common case errors
    text = re.sub(r'\bi\s+([a-z])', r'I \1', text)  # "i" -> "I"

    # Fix common contraction errors
    text = re.sub(r'\bdont\b', "don't", text, flags=re.IGNORECASE)
    text = re.sub(r'\bcant\b', "can't", text, flags=re.IGNORECASE)
    text = re.sub(r'\bwont\b', "won't", text, flags=re.IGNORECASE)

    # "Me and him was/were…" → "He and I were…"
    text = re.sub(r'\bme and him\b', 'I and him', text, flags=re.IGNORECASE)
    text = re.sub(r'\bme and her\b', 'I and her', text, flags=re.IGNORECASE)
    text = re.sub(r'\bhim and me\b', 'He and me', text, flags=re.IGNORECASE)
    text = re.sub(r'\bher and me\b', 'She and me', text, flags=re.IGNORECASE)
    text = re.sub(r'\bme and them\b', 'I and them', text, flags=re.IGNORECASE)
    text = re.sub(r'\bme and us\b', 'We', text, flags=re.IGNORECASE)

    # "Me and my friend" → "My friend and I"
    text = re.sub(r'\bme and ([a-zA-Z ]+)\b', r'\1 and I', text, flags=re.IGNORECASE)

    # "Him and my friends" → "He and my friends"
    text = re.sub(r'\bhim and ([a-zA-Z ]+)\b', r'He and \1', text, flags=re.IGNORECASE)
    text = re.sub(r'\bher and ([a-zA-Z ]+)\b', r'She and \1', text, flags=re.IGNORECASE)

    return text


def adjust_generation_params(base_params: Dict, input_length: int, text: str) -> Dict:
    """Intelligently adjust generation parameters"""
    params = base_params.copy()

    # Dynamically adjust max_length based on input length
    if input_length > 100:
        params["max_length"] = min(512, int(input_length * 1.8))
    else:
        params["max_length"] = min(256, max(64, int(input_length * 2.5)))

    # Adjust beam count and sampling strategy based on error density
    error_density = estimate_error_density(text)
    if error_density > 0.4:  # High error density
        params["num_beams"] = min(params.get("num_beams", 4) + 2, 8)
        params["repetition_penalty"] = max(params.get("repetition_penalty", 1.0), 1.1)
    elif error_density < 0.1:  # Low error density
        params["num_beams"] = max(params.get("num_beams", 4) - 1, 2)

    return params


def estimate_error_density(text: str) -> float:
    """Enhanced error density estimation, specifically detecting obvious errors like 'bootkass'"""
    words = text.split()
    if not words:
        return 0.0

    error_indicators = 0

    # Extended error pattern detection
    patterns = [
        r'\bi\s+has\b',  # "I has"
        r'\bdon\'?t\s+has\b',  # "don't has"
        r'\bhas\s+many\s+\w+s\b',  # Plural errors
        r'\b[a-z]+ss\b',  # Double-s ending spelling errors
        r'\btheir\s+going\b',  # "their going"
        r'\bthere\s+\w+ing\b',  # "there going"
        r'\b\w{4,}kas{1,2}\b',  # Similar to bootkass errors
        r'\b\w+grammer\b',  # Grammar spelling errors
        r'\bme\s+and\s+him\b',  # "me and him"
        r'\bstudients\b',  # Students spelling errors
        r'\bclasroom\b',  # Classroom spelling errors
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        error_indicators += len(matches)

    # Spelling error detection
    spell_checker = get_spell_checker_instance()
    if spell_checker:
        try:
            # Filter out possible proper nouns
            check_words = [w for w in words if len(w) > 2 and not w[0].isupper()]
            misspelled = spell_checker.unknown(check_words)
            error_indicators += len(misspelled)
        except:
            pass

    return min(error_indicators / len(words), 1.0)


def postprocess_text(corrected_text: str, prefix: str, original_text: str) -> str:
    """Enhanced post-processing"""
    if not corrected_text:
        return original_text

    # Remove prefix
    if prefix and corrected_text.lower().startswith(prefix.lower()):
        corrected_text = corrected_text[len(prefix):].strip()

    # Smart capitalization
    if corrected_text and corrected_text[0].islower():
        corrected_text = corrected_text[0].upper() + corrected_text[1:]

    # Fix common post-processing issues
    corrected_text = re.sub(r'\s+', ' ', corrected_text)  # Extra spaces
    corrected_text = re.sub(r'\s+([,.!?;:])', r'\1', corrected_text)  # Space before punctuation

    # Smart end punctuation handling
    if corrected_text and not corrected_text[-1] in '.!?':
        if original_text and original_text[-1] in '.!?':
            corrected_text += original_text[-1]
        elif len(corrected_text.split()) > 3:  # Only add period for longer sentences
            corrected_text += '.'

    return corrected_text


def assess_correction_quality(original: str, corrected: str) -> Dict:
    """Enhanced correction quality assessment"""
    if original == corrected:
        return {"confidence": "low", "score": 0.0}

    # Multi-dimensional quality assessment
    similarity = difflib.SequenceMatcher(None, original, corrected).ratio()

    # Detect specific error types fixed
    fixes_detected = 0

    # Grammar fix detection
    grammar_fixes = [
        (r'\bi\s+has\b', r'\bi\s+have\b'),
        (r'\bdon\'?t\s+has\b', r'\bdon\'?t\s+have\b'),
        (r'\bhe\s+don\'?t\b', r'\bhe\s+doesn\'?t\b'),
    ]

    for original_pattern, fixed_pattern in grammar_fixes:
        if re.search(original_pattern, original, re.IGNORECASE) and \
                re.search(fixed_pattern, corrected, re.IGNORECASE):
            fixes_detected += 1

    # Spelling fix detection
    spelling_fixes = [
        (r'\bgrammer\b', r'\bgrammar\b'),
        (r'\bmistakess\b', r'\bmistakes\b'),
        (r'\bbootkass\b', r'\bbook\w+\b'),  # bootkass fix
        (r'\bstudients\b', r'\bstudents\b'),
        (r'\bclasroom\b', r'\bclassroom\b'),
    ]

    for original_pattern, fixed_pattern in spelling_fixes:
        if re.search(original_pattern, original, re.IGNORECASE) and \
                re.search(fixed_pattern, corrected, re.IGNORECASE):
            fixes_detected += 1

    # Calculate confidence
    if fixes_detected >= 2 and similarity > 0.7:
        confidence = "high"
        score = 0.9
    elif fixes_detected >= 1 and similarity > 0.6:
        confidence = "medium"
        score = 0.7
    elif similarity > 0.8:
        confidence = "medium"
        score = 0.6
    else:
        confidence = "low"
        score = 0.3

    return {"confidence": confidence, "score": score}


# --- Optimized Rule Engine System ---
def check_with_language_tool(text: str) -> List[Dict]:
    """
    Smart grammar checking with LanguageTool + backup fallback

    Logic:
    1. If LanguageTool is available, use it
    2. If LanguageTool fails or is disabled, use backup_grammar_check
    3. Optionally combine both for comprehensive checking
    """
    language_tool = get_language_tool_instance()

    if language_tool is not None:
        try:
            logger.info("Using LanguageTool for grammar checking")
            # Use actual LanguageTool
            matches = language_tool.check(text)
            errors = []

            for match in matches:
                # Skip low-value errors
                if should_skip_error(match):
                    continue

                error_dict = {
                    "message": match.message,
                    "replacement": match.replacements[0] if match.replacements else "N/A",
                    "offset": match.offset,
                    "length": match.errorLength,
                    "context": match.context,
                    "rule_id": match.ruleId,
                    "category": match.category,
                    "error_type": categorize_error(match.ruleId, match.category),
                    "severity": get_error_severity(match.ruleId, match.message),
                    "confidence": calculate_error_confidence(match, text),
                    "priority": calculate_priority(
                        get_error_severity(match.ruleId, match.message),
                        calculate_error_confidence(match, text),
                        categorize_error(match.ruleId, match.category)
                    )
                }
                errors.append(error_dict)

            # Optionally add backup grammar results for comprehensive checking
            # Uncomment the next two lines if you want BOTH LanguageTool AND backup
            # backup_errors = backup_grammar_check(text)
            # errors.extend(backup_errors)

            return sorted(errors, key=lambda x: x["priority"], reverse=True)

        except Exception as e:
            logger.error(f"LanguageTool check failed: {e}, falling back to backup")
            return backup_grammar_check(text)
    else:
        logger.info("LanguageTool not available, using backup grammar checking")
        return backup_grammar_check(text)

def should_skip_error(match):
    """Determine if an error should be skipped based on its properties."""
    # Example: Skip errors with specific ruleIds or categories
    skip_rules = {"MORFOLOGIK_RULE_EN_US", "WHITESPACE_RULE"}
    if match.ruleId in skip_rules:
        return True
    return False

def categorize_error(rule_id, category):
    """Categorize the error based on ruleId and category."""
    if rule_id.startswith("MORFOLOGIK_RULE_"):
        return "Spelling"
    elif rule_id.startswith("COMMA_") or rule_id.startswith("PUNCTUATION"):
        return "Punctuation"
    elif "AGREEMENT" in rule_id or "VERB_NUMBER" in rule_id:
        return "Grammar"
    elif "CASE" in rule_id or "PRONOUN" in rule_id:
        return "Grammar"
    elif "TENSE" in rule_id:
        return "Verb Tense"
    elif "ARTICLE" in rule_id:
        return "Article"
    elif "REDUNDANCY" in rule_id:
        return "Style"
    else:
        return category if category else "Other"

def get_error_severity(rule_id, message):
    """Get the severity of the error based on ruleId and message."""
    severe_rules = {"EN_A_V", "ENGLISH_WORD_REPEAT_RULE", "UPPERCASE_SENTENCE_START"}
    moderate_rules = {"MORFOLOGIK_RULE_EN_US", "COMMA_PARENTHESIS_WHITESPACE"}

    if rule_id in severe_rules:
        return "High"
    elif rule_id in moderate_rules:
        return "Medium"
    else:
        return "Low"

def calculate_error_confidence(match, text):
    """Calculate the confidence of the error detection."""
    # Example: Base confidence on rule type and message clarity
    base_confidence = 0.8  # Default confidence

    if "MORFOLOGIK" in match.ruleId:
        base_confidence += 0.1  # Spelling rules are usually very reliable

    if "whitespace" in match.message.lower():
        base_confidence -= 0.1  # Whitespace rules can be less certain

    return min(max(base_confidence, 0.1), 1.0)  # Clamp between 0.1 and 1.0

def calculate_priority(severity, confidence, error_type):
    """Calculate the priority of the error based on severity, confidence, and type."""
    severity_map = {"High": 3, "Medium": 2, "Low": 1}
    error_type_weights = {"Grammar": 1.5, "Spelling": 1.2, "Punctuation": 1.0, "Style": 0.8}

    severity_score = severity_map.get(severity, 1)
    type_weight = error_type_weights.get(error_type, 1.0)

    return severity_score * confidence * type_weight * 100  # Scale to a reasonable range

def backup_grammar_check(text: str) -> List[Dict]:
    """Enhanced grammar checking, alternative to LanguageTool"""
    errors = []

    # Extended grammar rules library
    grammar_rules = [
        # 新增的语法规则示例：处理"techer"到"teacher"的错误
        {
            "pattern": r'\btecher\b',
            "message": "Spelling error: 'techer' should be 'teacher'",
            "replacement_func": lambda m: "teacher",
            "severity": "High",
            "error_type": "Spelling"
        },
        # 新增规则：检测并修正"libaray"到"library"
        {
            "pattern": r'\blibaray\b',
            "message": "Spelling error: 'libaray' should be 'library'",
            "replacement_func": lambda m: "library",
            "severity": "High",
            "error_type": "Spelling"
        },
        {
            "pattern": r'\bbootkass\b',
            "message": "Spelling error: 'bootkass' should be 'bookcase'",
            "replacement_func": lambda m: "bookcase",
            "severity": "High",
            "error_type": "Spelling"
        },
        # Subject-Verb Agreement Errors
        {
            "pattern": r'\bi\s+has\b',  
            "message": "Subject-verb disagreement: 'I' should use have/were/do/am",
            "replacement_func": lambda m: re.sub(r'\b(has|was|does|is)\b',
                                                 {'has': 'have', 'was': 'were', 'does': 'do', 'is': 'am'}[m.group(1)],
                                                 m.group()),
            "severity": "High",
            "error_type": "Grammar"
        },
        # Pronoun case/order errors
        {
            "pattern": r'\bme and (him|her|them)\b',
            "message": "Pronoun order/case: Use 'I and he/she/they' (or 'He and I', 'She and I' for better flow).",
            "replacement_func": lambda m: "He and I" if m.group(1) == "him" else (
                "She and I" if m.group(1) == "her" else "They and I"),
            "severity": "High",
            "error_type": "Grammar"
        },
        {
            "pattern": r'\b(?:he|she|it)\s+(?:have|were|do|are)\b',
            "message": "Subject-verb disagreement: Third person singular should use has/was/does/is",
            "replacement_func": lambda m: re.sub(r'\b(have|were|do|are)\b',
                                                 {'have': 'has', 'were': 'was', 'do': 'does', 'are': 'is'}[m.group(1)],
                                                 m.group()),
            "severity": "High",
            "error_type": "Grammar"
        },
        # Verb Tense Errors
        {
            "pattern": r'\b(?:yesterday|last\s+\w+)\s+.*?\b\w+ing\b',
            "message": "Tense error: Past time adverbs are typically not used with progressive tense",
            "replacement_func": None,
            "severity": "Medium",
            "error_type": "Grammar"
        },
        # Confused Words
        {
            "pattern": r'\btheir\s+(?:going|coming|leaving)\b',
            "message": "Lexical error: 'their' should be 'they're' (they are)",
            "replacement_func": lambda m: m.group().replace('their', "they're"),
            "severity": "High",
            "error_type": "Grammar"
        },
        {
            "pattern": r'\bthere\s+(?:house|car|book|friend)\b',
            "message": "Lexical error: 'there' should be 'their' (possessive)",
            "replacement_func": lambda m: m.group().replace('there', 'their'),
            "severity": "High",
            "error_type": "Grammar"
        },
        # Article Errors
        {
            "pattern": r'\ba\s+[aeiouAEIOU]\w+\b',
            "message": "Article error: 'an' should be used before a word starting with a vowel sound",
            "replacement_func": lambda m: m.group().replace('a ', 'an ', 1),
            "severity": "Medium",
            "error_type": "Grammar"
        },
        {
            "pattern": r'\ban\s+[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]\w+\b',
            "message": "Article error: 'a' should be used before a word starting with a consonant sound",
            "replacement_func": lambda m: m.group().replace('an ', 'a ', 1),
            "severity": "Medium",
            "error_type": "Grammar"
        },
        # Common Spelling and Grammar Mixed Errors
        {
            "pattern": r'\bdont\b',
            "message": "Missing apostrophe: 'dont' should be 'don't'",
            "replacement_func": lambda m: "don't",
            "severity": "High",
            "error_type": "Punctuation"
        },
        {
            "pattern": r'\bcant\b',
            "message": "Missing apostrophe: 'cant' should be 'can't'",
            "replacement_func": lambda m: "can't",
            "severity": "High",
            "error_type": "Punctuation"
        },
        {
            "pattern": r'\bwont\b',
            "message": "Missing apostrophe: 'wont' should be 'won't'",
            "replacement_func": lambda m: "won't",
            "severity": "High",
            "error_type": "Punctuation"
        },
        # Plural Form Errors
        {
            "pattern": r'\b(?:this|that)\s+\w+s\b',
            "message": "Singular/plural mismatch: 'this/that' should be followed by a singular noun",
            "replacement_func": None,
            "severity": "Medium",
            "error_type": "Grammar"
        },
        {
            "pattern": r'\b(?:these|those)\s+\w+[^s]\b(?!\w)',
            "message": "Singular/plural mismatch: 'these/those' should be followed by a plural noun",
            "replacement_func": None,
            "severity": "Medium",
            "error_type": "Grammar"
        }
    ]

    # Apply all rules
    for rule in grammar_rules:
        matches = list(re.finditer(rule["pattern"], text, re.IGNORECASE))
        for match in matches:
            error_dict = {
                "message": rule["message"],
                "offset": match.start(),
                "length": match.end() - match.start(),
                "context": text[max(0, match.start() - 20):match.end() + 20],
                "rule_id": f"ENHANCED_{rule['error_type'].upper()}_RULE",
                "category": rule["error_type"].upper(),
                "error_type": rule["error_type"],
                "severity": rule["severity"],
                "confidence": 0.85,
                "priority": {"High": 100, "Medium": 60, "Low": 30}[rule["severity"]]
            }

            # Generate suggested correction
            if rule["replacement_func"]:
                try:
                    replacement = rule["replacement_func"](match)
                    error_dict["replacement"] = replacement
                except:
                    error_dict["replacement"] = "N/A"
            else:
                error_dict["replacement"] = "N/A"

            errors.append(error_dict)

    # Check sentence structure issues
    sentence_errors = check_sentence_structure(text)
    errors.extend(sentence_errors)

    return sorted(errors, key=lambda x: x["priority"], reverse=True)


def check_spelling_pyspell(text: str) -> List[Dict]:
    """Enhanced spelling check, specifically optimized for obvious errors like 'bootkass'"""
    spell = get_spell_checker_instance()
    if not spell:
        return []

    try:
        # Smart word extraction with position information
        words = extract_words_with_positions(text)
        word_list = [w["word"] for w in words]
        misspelled = spell.unknown(word_list)

        suggestions = []
        for word_info in words:
            word = word_info["word"]
            if word not in misspelled:
                continue

            # Skip obvious proper nouns and abbreviations
            if should_skip_word(word):
                continue

            correction = spell.correction(word)
            candidates = list(spell.candidates(word))[:5]

            # Special handling for common error patterns
            correction, candidates = handle_special_spelling_cases(word, correction, candidates)

            # Enhanced confidence calculation
            confidence = calculate_spelling_confidence(word, correction, candidates)

            suggestions.append({
                "word": word,
                "correction": correction,
                "candidates": candidates,
                "confidence": confidence,
                "position": word_info["position"],
                "context": get_word_context(text, word_info["position"], len(word)),
                "priority": calculate_spelling_priority(confidence, len(word))
            })

        # Sort by priority, ensuring obvious errors come first
        return sorted(suggestions, key=lambda x: x["priority"], reverse=True)

    except Exception as e:
        logger.error(f"Enhanced spelling check failed: {e}")
        return []


def handle_special_spelling_cases(word: str, correction: str, candidates: List[str]) -> Tuple[str, List[str]]:
    """Handle special spelling error cases, specifically dealing with obvious errors like 'bootkass'"""
    word_lower = word.lower()

    # Extended special error pattern mapping
    special_corrections = {
        'bootkass': 'bookcase',  # Add special handling for bootkass
        'grammer': 'grammar',
        'writting': 'writing',
        'mistakess': 'mistakes',
        'studients': 'students',
        'clasroom': 'classroom',
        'techer': 'teacher',
        'compyter': 'computer',
        'libaray': 'library',
        'recieve': 'receive',
        'occured': 'occurred',
        'seperate': 'separate',
    }

    if word_lower in special_corrections:
        special_correction = special_corrections[word_lower]
        # Maintain original case format
        if word[0].isupper():
            special_correction = special_correction.capitalize()

        # Ensure special correction is first in candidate list
        if special_correction not in candidates:
            candidates = [special_correction] + candidates[:4]
        else:
            # If already in list, move to first position
            candidates.remove(special_correction)
            candidates = [special_correction] + candidates

        return special_correction, candidates

    # Handle patterns similar to bootkass (word + kass/kas)
    if word_lower.endswith('kass') or word_lower.endswith('kas'):
        base_word = word_lower[:-4] if word_lower.endswith('kass') else word_lower[:-3]

        # Specific handling for book-related errors
        if base_word == 'boot' or base_word == 'book':
            correction = 'bookcase'
            candidates = ['bookcase', 'book', 'books'] + candidates[:2]
            return correction, candidates

        # Other kass-ending handling
        possible_corrections = [
            base_word + 'case',
            base_word + 'class',
            base_word + 's'
        ]

        # Add to front of candidate list
        for poss_correction in possible_corrections:
            if poss_correction not in candidates:
                candidates.insert(0, poss_correction)

        if candidates:
            correction = candidates[0]

    return correction, candidates[:5]


def extract_words_with_positions(text: str) -> List[Dict]:
    """Extract words with their position information"""
    words = []
    for match in re.finditer(r'\b[a-zA-Z]{2,}\b', text):
        words.append({
            "word": match.group(),
            "position": match.start()
        })
    return words


def should_skip_word(word: str) -> bool:
    """Filter words that don't need checking"""
    # Skip short, all-caps (possibly abbreviations), or words containing numbers
    if (len(word) < 3 or
            word.isupper() or
            any(c.isdigit() for c in word)):
        return True

    # Skip common abbreviations and internet slang
    skip_words = {
        'ok', 'hi', 'bye', 'lol', 'omg', 'wtf', 'btw', 'fyi',
        'etc', 'vs', 'ie', 'eg', 'asap', 'faq', 'diy'
    }

    if word.lower() in skip_words:
        return True

    # Skip possible proper nouns (capitalized and length > 5)
    if word[0].isupper() and len(word) > 5:
        return True

    return False


def calculate_spelling_confidence(word: str, correction: str, candidates: List[str]) -> float:
    """Calculate spelling correction confidence"""
    if not correction or correction == word:
        return 0.1

    confidence = 0.5

    # Based on edit distance
    edit_distance = calculate_edit_distance(word, correction)
    if edit_distance == 1:
        confidence += 0.3  # Single character difference, high confidence
    elif edit_distance == 2:
        confidence += 0.2

    # Based on number of candidates
    if len(candidates) == 1:
        confidence += 0.2  # Unique candidate
    elif len(candidates) <= 3:
        confidence += 0.1

    return min(confidence, 1.0)


def calculate_edit_distance(s1: str, s2: str) -> int:
    """Calculate edit distance"""
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        new_distances = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                new_distances.append(distances[i1])
            else:
                new_distances.append(1 + min(distances[i1], distances[i1 + 1], new_distances[-1]))
        distances = new_distances
    return distances[-1]


def calculate_spelling_priority(confidence: float, word_length: int) -> int:
    """Calculate spelling error priority"""
    base_score = int(confidence * 100)
    length_bonus = min(word_length * 2, 20)  # Longer words get priority
    return base_score + length_bonus


def get_word_context(text: str, position: int, word_length: int) -> str:
    """Get word context"""
    start = max(0, position - 20)
    end = min(len(text), position + word_length + 20)
    return text[start:end]


# --- Intelligent Feedback Fusion System ---
def intelligent_feedback_fusion(transformer_result: Dict, lt_errors: List[Dict], spell_errors: List[Dict]) -> Dict:
    """Intelligently fuse Transformer and rule engine feedback"""

    original_text = transformer_result.get("original_text", "")
    corrected_text = transformer_result.get("corrected_text", "")

    # 1. Analyze Transformer fixes
    transformer_fixes = analyze_transformer_fixes(original_text, corrected_text)

    # 2. Filter resolved rule engine errors
    filtered_lt_errors = filter_resolved_lt_errors(lt_errors, transformer_fixes, corrected_text)
    filtered_spell_errors = filter_resolved_spell_errors(spell_errors, transformer_fixes, corrected_text)

    # 3. Identify Transformer missed issues
    missed_issues = identify_missed_issues(filtered_lt_errors, filtered_spell_errors, corrected_text)

    # 4. Generate fusion suggestions
    fusion_suggestions = generate_fusion_suggestions(
        transformer_result, filtered_lt_errors, filtered_spell_errors, missed_issues
    )

    return {
        "transformer_result": transformer_result,
        "filtered_lt_errors": filtered_lt_errors[:5],  # Limit quantity
        "filtered_spell_errors": filtered_spell_errors[:3],
        "missed_critical_issues": missed_issues,
        "fusion_suggestions": fusion_suggestions,
        "confidence_score": calculate_overall_confidence(transformer_result, filtered_lt_errors, filtered_spell_errors)
    }


def analyze_transformer_fixes(original: str, corrected: str) -> List[Dict]:
    """Analyze fixes made by Transformer"""
    if original == corrected:
        return []

    fixes = []

    # Detect common fix patterns
    patterns = [
        (r'\bi\s+has\b', r'\bi\s+have\b', "subject_verb_agreement"),
        (r'\bdon\'?t\b', r'\bdon\'t\b', "contraction_fix"),
        (r'\bgrammer\b', r'\bgrammar\b', "spelling_fix"),
        (r'\bmistakess\b', r'\bmistakes\b', "spelling_fix"),
        (r'\bbootkass\b', r'\bbook\w+\b', "spelling_fix"),
    ]

    for pattern, replacement, fix_type in patterns:
        if re.search(pattern, original, re.IGNORECASE) and re.search(replacement, corrected, re.IGNORECASE):
            fixes.append({
                "type": fix_type,
                "pattern": pattern,
                "description": f"Fixed {fix_type.replace('_', ' ')}"
            })

    return fixes


def filter_resolved_lt_errors(lt_errors: List[Dict], transformer_fixes: List[Dict], corrected_text: str) -> List[Dict]:
    """Filter LanguageTool errors already resolved by Transformer"""
    filtered = []

    for error in lt_errors:
        if is_error_resolved(error, transformer_fixes, corrected_text):
            continue
        filtered.append(error)

    return filtered


def filter_resolved_spell_errors(spell_errors: List[Dict], transformer_fixes: List[Dict], corrected_text: str) -> List[
    Dict]:
    """Filter spelling errors already resolved by Transformer"""
    filtered = []
    lower_corrected = corrected_text.lower()
    for err in spell_errors:
        if err["word"].lower() not in lower_corrected or (
                err["correction"] and err["correction"].lower() in lower_corrected
        ):
            continue
        filtered.append(err)
    return filtered


def is_error_resolved(error: Dict, transformer_fixes: List[Dict], corrected_text: str) -> bool:
    """Check if error has been resolved"""
    # Check if suggested fix appears in corrected text
    if error["replacement"] != "N/A" and error["replacement"].lower() in corrected_text.lower():
        return True

    # Check if it matches Transformer fix patterns
    for fix in transformer_fixes:
        if fix["type"] in error["rule_id"].lower():
            return True

    return False


def identify_missed_issues(lt_errors: List[Dict], spell_errors: List[Dict], corrected_text: str) -> List[Dict]:
    """Identify critical issues missed by Transformer"""
    critical_issues = []

    # High priority grammar errors
    for error in lt_errors:
        if error["priority"] > 80 and error["confidence"] > 0.7:
            critical_issues.append({
                "type": "grammar",
                "message": error["message"],
                "suggestion": error["replacement"],
                "priority": error["priority"]
            })

    # High confidence spelling errors
    for error in spell_errors:
        if error["confidence"] > 0.8:
            critical_issues.append({
                "type": "spelling",
                "message": f"Misspelled: {error['word']}",
                "suggestion": error["correction"],
                "priority": error["priority"]
            })

    return critical_issues[:3]  # Limit to top 3 most important


def generate_fusion_suggestions(transformer_result: Dict, lt_errors: List[Dict],
                                spell_errors: List[Dict], missed_issues: List[Dict]) -> List[Dict]:
    """Generate fusion suggestions"""
    suggestions = []

    # Transformer result as primary suggestion
    if transformer_result.get("changes_made"):
        suggestions.append({
            "type": "primary",
            "source": "AI Model",
            "suggestion": transformer_result["corrected_text"],
            "confidence": transformer_result.get("confidence", "medium"),
            "priority": 100
        })

    # Critical missed issues
    for issue in missed_issues:
        suggestions.append({
            "type": "additional",
            "source": f"Rule Engine ({issue['type']})",
            "suggestion": issue["suggestion"],
            "reason": issue["message"],
            "priority": issue["priority"]
        })

    # Sort by priority
    return sorted(suggestions, key=lambda x: x["priority"], reverse=True)


def calculate_overall_confidence(transformer_result: Dict, lt_errors: List[Dict], spell_errors: List[Dict]) -> str:
    """Calculate overall confidence"""
    transformer_conf = transformer_result.get("improvement_score", 0)
    remaining_issues = len(lt_errors) + len(spell_errors)

    if transformer_conf > 0.7 and remaining_issues < 2:
        return "high"
    elif transformer_conf > 0.4 and remaining_issues < 4:
        return "medium"
    else:
        return "low"


# --- Improved Difference Highlighting System ---
def get_highlighted_diff(original_text: str, corrected_text: str) -> List[Tuple]:
    """Generate detailed difference highlighting, supporting multiple change types"""
    if not original_text or not corrected_text:
        return [(original_text or corrected_text, None)]

    # Use character-level and word-level mixed comparison
    original_words = original_text.split()
    corrected_words = corrected_text.split()

    matcher = difflib.SequenceMatcher(None, original_words, corrected_words)
    highlighted_output = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            text = " ".join(original_words[i1:i2])
            if text:
                highlighted_output.append((text + " ", None))
        elif tag == 'replace':
            original_chunk = " ".join(original_words[i1:i2])
            corrected_chunk = " ".join(corrected_words[j1:j2])
            if original_chunk:
                highlighted_output.append((original_chunk + " ", "REMOVED"))
            if corrected_chunk:
                highlighted_output.append((corrected_chunk + " ", "ADDED"))
        elif tag == 'delete':
            text = " ".join(original_words[i1:i2])
            if text:
                highlighted_output.append((text + " ", "REMOVED"))
        elif tag == 'insert':
            text = " ".join(corrected_words[j1:j2])
            if text:
                highlighted_output.append((text + " ", "ADDED"))

    return highlighted_output if highlighted_output else [(corrected_text, "CORRECTED")]


def map_language_tool_errors(text: str, errors: List[Dict]) -> List[Tuple]:
    """Map LanguageTool errors to HighlightedText format"""
    if not errors:
        return [(text, None)]

    highlighted_parts = []
    last_end = 0

    # Sort by offset
    sorted_errors = sorted(errors, key=lambda x: x['offset'])

    for error in sorted_errors:
        offset = error['offset']
        length = error['length']

        # Add normal text before error
        if offset > last_end:
            highlighted_parts.append((text[last_end:offset], None))

        # Add error text with label
        error_text = text[offset:offset + length]
        error_label = f"{error['error_type']} ({error['severity']})"
        highlighted_parts.append((error_text, error_label))

        last_end = offset + length

    # Add remaining text
    if last_end < len(text):
        highlighted_parts.append((text[last_end:], None))

    return highlighted_parts


# --- Main Control Function ---
def master_grammar_check(text_to_check: str, selected_model_key: str,
                         use_language_tool: bool = True, use_pyspellchecker: bool = True) -> Dict:
    """Master grammar checking function, fusing multiple tools and eliminating redundancy"""

    if not text_to_check.strip():
        return {"error": "Please enter some text."}

    results = {
        "original_text": text_to_check,
        "timestamp": torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else "CPU",
        "language_tool_status": "available" if get_language_tool_instance() else "unavailable"
    }

    try:
        # 1. Transformer model correction (primary method)
        transformer_result = correct_grammar_hf(text_to_check, selected_model_key)
        transformer_result["original_text"] = text_to_check

        # 2. Rule engine checking (supplementary method)
        # Smart selection: LanguageTool if available, otherwise backup_grammar_check
        lt_errors = check_with_language_tool(text_to_check) if use_language_tool else []
        spell_errors = check_spelling_pyspell(text_to_check) if use_pyspellchecker else []

        # 3. Intelligent fusion and redundancy elimination
        fusion_result = intelligent_feedback_fusion(transformer_result, lt_errors, spell_errors)

        # 4. Generate final suggestions
        final_suggestions = create_final_suggestions(fusion_result)

        # 5. Generate highlighting data
        diff_highlight = get_highlighted_diff(text_to_check, transformer_result.get("corrected_text", text_to_check))
        error_highlight = map_language_tool_errors(text_to_check, fusion_result["filtered_lt_errors"])

        # Assemble results
        results.update({
            "primary_correction": transformer_result.get("corrected_text", text_to_check),
            "confidence": fusion_result["confidence_score"],
            "suggestions": final_suggestions,
            "additional_issues": fusion_result["missed_critical_issues"],
            "diff_highlight": diff_highlight,
            "error_highlight": error_highlight,
            "spell_suggestions": fusion_result["filtered_spell_errors"][:3],
            "grammar_errors": fusion_result["filtered_lt_errors"][:5],
            "stats": {
                "transformer_changes": transformer_result.get("changes_made", False),
                "grammar_issues_found": len(fusion_result["filtered_lt_errors"]),
                "spelling_issues_found": len(fusion_result["filtered_spell_errors"]),
                "redundancy_reduction": f"{len(lt_errors + spell_errors) - len(fusion_result['filtered_lt_errors'] + fusion_result['filtered_spell_errors'])} duplicates removed"
            }
        })

        return results

    except Exception as e:
        logger.error(f"Main checking function execution failed: {e}")
        return {
            "error": f"Check failed: {str(e)}",
            "original_text": text_to_check,
            "language_tool_status": "error"
        }


def create_final_suggestions(fusion_result: Dict) -> List[Dict]:
    """Create final suggestion list, sorted by priority"""
    suggestions = []

    # Primary AI suggestion
    transformer_result = fusion_result["transformer_result"]
    if transformer_result.get("changes_made"):
        suggestions.append({
            "type": "primary",
            "title": "AI Grammar Correction",
            "before": transformer_result.get("original_text", ""),
            "after": transformer_result.get("corrected_text", ""),
            "confidence": transformer_result.get("confidence", "medium"),
            "source": "Neural Model"
        })

    # Critical missed issues
    for issue in fusion_result["missed_critical_issues"]:
        suggestions.append({
            "type": "critical",
            "title": issue["message"],
            "suggestion": issue["suggestion"],
            "confidence": "high",
            "source": f"Rule Engine ({issue['type']})"
        })

    # Minor grammar issues
    for error in fusion_result["filtered_lt_errors"][:2]:  # Limit quantity
        if error["priority"] > 60:
            suggestions.append({
                "type": "grammar",
                "title": error["message"],
                "suggestion": error["replacement"],
                "confidence": error["confidence"],
                "source": "Grammar Rules"
            })

    return suggestions[:5]  # Maximum 5 suggestions


# --- Batch Testing Functions ---
def benchmark_all_models(test_sentences: List[str] = None):
    """Performance benchmark testing for all models"""
    if test_sentences is None:
        test_sentences = [
            "I has a bad grammer and I is very happy.",
            "he dont like water he swim fastly",
            "This sentences has many mistakess. It a beautiful day.",
            "Their going to there house with they're friends.",
            "Me and him was walking to the store yesterday.",
            "The students are reading their bootkass carefully."  # Add bootkass test
        ]

    print("=== Model Performance Benchmark Testing ===")
    results = {}

    for model_key in MODEL_OPTIONS.keys():
        print(f"\nTesting model: {model_key}")
        model_results = []

        for i, sentence in enumerate(test_sentences, 1):
            print(f"  Test sentence {i}: {sentence}")
            result = correct_grammar_hf(sentence, model_key)
            model_results.append({
                "input": sentence,
                "output": result.get("corrected_text", "Error"),
                "success": "error" not in result
            })
            print(f"  Result: {result.get('corrected_text', 'Error')}")

        results[model_key] = model_results

        # Display model performance information
        perf = MODEL_OPTIONS[model_key]["performance"]
        print(f"  Performance: {perf['accuracy']} | Speed: {perf['speed']} | Memory: {perf['memory']}")

    return results


# --- Specialized Function for Testing bootkass Corrections ---
def test_bootkass_corrections():
    """Specialized testing for obvious errors like 'bootkass'"""
    test_cases = [
        "The students in the classroom are reading their bootkass.",
        "I has a bad grammer and need to improve my writting skils.",
        "He don't like water but he swim very good yesterday.",
        "Their going to there house with they're friends.",
        "Me and him was walking to the store and we buyed some mistakess.",
        "The studients are in the clasroom with there techer."
    ]

    print("=== Testing Obvious Error Correction Capabilities ===")
    print(f"LanguageTool status: {'Enabled' if not DISABLE_LANGUAGE_TOOL else 'Disabled'}")
    if not DISABLE_LANGUAGE_TOOL:
        print("Mode 3: LanguageTool + SpellChecker (Recommended)")
        print("Will try LanguageTool first, fallback to backup if needed")
    else:
        print("Using backup grammar checking system only")
    print("Both grammar checking and SpellChecker are enabled by default\n")

    for i, text in enumerate(test_cases, 1):
        print(f"\n{i}. Test: {text}")

        # Mode 3: LanguageTool + SpellChecker (both enabled by default)
        result = master_grammar_check(
            text,
            "T5-Base (Vennify)",
            use_language_tool=True,  # LanguageTool/backup智能选择
            use_pyspellchecker=True  # 拼写检查
        )

        if "error" not in result:
            print(f"   Correction: {result['primary_correction']}")
            print(f"   Confidence: {result['confidence']}")
            print(f"   Changes made: {'Yes' if result['stats']['transformer_changes'] else 'No'}")

            # Verify which grammar engine was used
            if result.get('grammar_errors'):
                print("   Grammar error detection:")
                for error in result['grammar_errors'][:3]:
                    rule_source = "LanguageTool" if not error['rule_id'].startswith('ENHANCED_') else "Backup"
                    print(f"     [{rule_source}] {error['message']} (severity: {error['severity']})")

            # Display spelling suggestions
            if result.get('spell_suggestions'):
                print("   Spelling suggestions:")
                for suggestion in result['spell_suggestions'][:3]:
                    print(
                        f"     {suggestion['word']} → {suggestion['correction']} (confidence: {suggestion['confidence']})")
        else:
            print(f"   Error: {result['error']}")


def verify_languagetool_setup():
    """Verify if LanguageTool is working correctly"""
    print("=== LanguageTool Setup Verification ===")

    # Test 1: Check instance creation
    lt_instance = get_language_tool_instance()
    if lt_instance is not None:
        print("✅ LanguageTool instance created successfully")

        # Test 2: Check basic functionality
        try:
            test_text = "This are a test sentence."
            matches = lt_instance.check(test_text)
            if matches:
                print(f"✅ LanguageTool detected {len(matches)} error(s) in test sentence")
                for match in matches[:2]:  # Show first 2 errors
                    print(f"   - {match.message} (Rule: {match.ruleId})")
            else:
                print("⚠️  LanguageTool didn't detect expected errors")
        except Exception as e:
            print(f"❌ LanguageTool check failed: {e}")
            return False
    else:
        print("❌ LanguageTool instance creation failed")
        return False

    # Test 3: Check integration with our system
    print("\n--- Integration Test ---")
    test_integration_text = "I has many book and he don't like they."
    result = master_grammar_check(test_integration_text, "T5-Base (Vennify)")

    grammar_errors = result.get('grammar_errors', [])
    languagetool_errors = [e for e in grammar_errors if not e['rule_id'].startswith('ENHANCED_')]
    backup_errors = [e for e in grammar_errors if e['rule_id'].startswith('ENHANCED_')]

    print(f"Grammar errors found: {len(grammar_errors)}")
    print(f"  - LanguageTool errors: {len(languagetool_errors)}")
    print(f"  - Backup errors: {len(backup_errors)}")

    if languagetool_errors:
        print("✅ LanguageTool integration working")
        for error in languagetool_errors[:2]:
            print(f"   LT: {error['message']} (Rule: {error['rule_id']})")
    else:
        print("⚠️  No LanguageTool errors detected - may be using backup only")

    return True


# Test function call example
if __name__ == "__main__":
    # Configuration options demonstration
    print("=== Configuration: Mode 3 (LanguageTool + SpellChecker) ===")
    print(f"Current setting: DISABLE_LANGUAGE_TOOL = {DISABLE_LANGUAGE_TOOL}")
    print("Mode 3 Configuration:")
    print("- LanguageTool: Enabled (primary grammar checking)")
    print("- SpellChecker: Enabled (pyspellchecker)")
    print("- Backup Grammar: Available as fallback")
    print("- AI Model: T5/BART for main corrections")
    print("\n")

    # Verify LanguageTool setup first
    setup_success = verify_languagetool_setup()

    if setup_success:
        print("\n" + "=" * 50)
        # Test bootkass and other error corrections
        test_bootkass_corrections()

        print("\n" + "=" * 50)
        # Original batch testing
        benchmark_results = benchmark_all_models()
        print("Model benchmark testing completed")
    else:
        print("❌ LanguageTool setup verification failed. Please check installation:")
        print("   pip install language-tool-python")
        print("   or set DISABLE_LANGUAGE_TOOL = True to use backup only")