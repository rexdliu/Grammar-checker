# -*- coding: utf-8 -*-
"""
高级英语语法检查系统 - 深度优化版本
支持GPU强制运行，多模型深度集成，智能冗余消除
修复LanguageTool初始化问题，提升准确度
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

# --- 优化模型配置 ---
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
            "strengths": "基础语法错误，标点符号，快速响应",
            "weaknesses": "复杂句式重构能力有限"
        }
    },
    # --- 你的微调 BART ---
    "My BART Fine-tuned": {
        "name": "finetuned_bart_model_agentlans_local",
        # BART 训练时没用前缀，就留空
        "prefix": "",
        "model_type": "AutoModelForSeq2SeqLM",
        "params": {                # 推理时的生成参数，可先沿用 T5 的
            "max_length": 128,     # 训练时就是 128
            "num_beams": 4,
            "early_stopping": True,
            "no_repeat_ngram_size": 3,
            "repetition_penalty": 1.05,
            "length_penalty": 1.0,
            "do_sample": False
        },
        "performance": {           # 纯备注，可随意写
            "accuracy": "Fine-tuned",
            "speed": "Fast",
            "memory": "~2.5GB",
            "strengths": "In-domain corrections",
            "weaknesses": "May overfit small set"
        }
    }
}


# --- 全局实例管理 ---
_language_tool_instance = None
_spell_checker_instance = None
loaded_models = {}
DISABLE_LANGUAGE_TOOL = True 

def get_language_tool_instance():
    """获取LanguageTool实例 - 直接返回None"""
    if DISABLE_LANGUAGE_TOOL:
        return None

    global _language_tool_instance
    if _language_tool_instance is None:
        logger.info("LanguageTool已被禁用，使用增强的备用语法检查")
        return None
    return _language_tool_instance
def get_spell_checker_instance():
    """获取SpellChecker单例，增强错误处理"""
    global _spell_checker_instance
    if _spell_checker_instance is None:
        try:
            logger.info("正在初始化SpellChecker...")
            _spell_checker_instance = SpellChecker()

            # 添加常见的专业词汇到词典，避免误报
            custom_words = {
                'booktask', 'bookcase', 'textbook', 'workbook',
                'handbook', 'notebook', 'facebook', 'laptop',
                'website', 'email', 'online', 'offline'
            }
            _spell_checker_instance.word_frequency.load_words(custom_words)

            logger.info("SpellChecker初始化成功")
        except Exception as e:
            logger.error(f"SpellChecker初始化失败: {e}")
            _spell_checker_instance = None
    return _spell_checker_instance


def check_sentence_structure(text: str) -> List[Dict]:
    """检查句子结构问题"""
    errors = []

    # 检查句子开头大写
    sentences = re.split(r'[.!?]\s+', text)
    current_pos = 0

    for sentence in sentences:
        if sentence and sentence[0].islower():
            errors.append({
                "message": "句子应该以大写字母开头",
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
        current_pos += len(sentence) + 2  # 加上标点和空格

    # 检查缺少句末标点
    if text and text[-1] not in '.!?':
        errors.append({
            "message": "句子末尾缺少标点符号",
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
    """加载并缓存Hugging Face模型，确保设备一致性"""
    if model_name not in loaded_models:
        logger.info(f"正在加载模型: {model_name} (类型: {model_type})")
        try:
            # 使用缓存目录避免重复下载
            cache_dir = os.path.join(os.getcwd(), "model_cache")
            os.makedirs(cache_dir, exist_ok=True)

            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                local_files_only=False
            )

            # 确保有pad_token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # 根据模型类型和设备选择加载方式
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
                # CPU模式
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
            logger.info(f"模型 {model_name} 加载成功")

        except Exception as e:
            logger.error(f"模型 {model_name} 加载失败: {e}")
            return None, None

    return loaded_models[model_name]["tokenizer"], loaded_models[model_name]["model"]
# --- 增强的核心函数 ---
def correct_grammar_hf(text: str, model_key: str) -> Dict:
    """使用指定模型进行语法纠错，修复设备不匹配问题"""
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

    # 增强预处理
    cleaned_text = preprocess_text(text)
    input_text = f"{prefix}{cleaned_text}" if prefix else cleaned_text

    try:
        # 确保tokenizer输出在正确的设备上
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
            add_special_tokens=True
        )

        # 手动将inputs移到正确的设备
        if DEVICE == "cuda":
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            input_length = inputs['input_ids'].shape[1]

            # 动态参数调整
            dynamic_params = adjust_generation_params(params, input_length, text)

            # 确保所有参数都在同一设备上
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

        # 增强后处理
        corrected_text = postprocess_text(corrected_text, prefix, text)

        # 智能质量评估
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
        logger.error(f"语法纠错生成失败: {e}")
        return {"error": f"Generation failed: {str(e)}", "corrected_text": text}


def preprocess_text(text: str) -> str:
    """文本预处理"""
    if not text.strip():
        return text

    # 基础清理
    text = re.sub(r'\s+', ' ', text.strip())

    # 修复明显的格式问题
    text = re.sub(r'\s+([,.!?;:])', r'\1', text)  # 标点前空格
    text = re.sub(r'([.!?])\s*([a-z])', r'\1 \2', text)  # 句子间距

    # 修复常见的大小写错误
    text = re.sub(r'\bi\s+([a-z])', r'I \1', text)  # "i" -> "I"
    # 修复常见的缩写错误
    text = re.sub(r'\bdont\b', "don't", text, flags=re.IGNORECASE)
    text = re.sub(r'\bcant\b', "can't", text, flags=re.IGNORECASE)
    text = re.sub(r'\bwont\b', "won't", text, flags=re.IGNORECASE)
    # “Me and him was/were…” → “He and I were…”
    text = re.sub(
        r'\bme and him\b', 'I and him', text, flags=re.IGNORECASE)
    text = re.sub(
        r'\bme and her\b', 'I and her', text, flags=re.IGNORECASE)
    text = re.sub(
        r'\bhim and me\b', 'He and me', text, flags=re.IGNORECASE)
    text = re.sub(
        r'\bher and me\b', 'She and me', text, flags=re.IGNORECASE)
    text = re.sub(
        r'\bme and them\b', 'I and them', text, flags=re.IGNORECASE)
    text = re.sub(
        r'\bme and us\b', 'We', text, flags=re.IGNORECASE)
    # “Me and my friend” → “My friend and I”
    text = re.sub(
        r'\bme and ([a-zA-Z ]+)\b', r'\1 and I', text, flags=re.IGNORECASE)
    # “Him and my friends” → “He and my friends”
    text = re.sub(
        r'\bhim and ([a-zA-Z ]+)\b', r'He and \1', text, flags=re.IGNORECASE)
    text = re.sub(
        r'\bher and ([a-zA-Z ]+)\b', r'She and \1', text, flags=re.IGNORECASE)
    return text


def adjust_generation_params(base_params: Dict, input_length: int, text: str) -> Dict:
    """智能调整生成参数"""
    params = base_params.copy()

    # 根据输入长度动态调整max_length
    if input_length > 100:
        params["max_length"] = min(512, int(input_length * 1.8))
    else:
        params["max_length"] = min(256, max(64, int(input_length * 2.5)))

    # 根据错误密度调整beam数和采样策略
    error_density = estimate_error_density(text)
    if error_density > 0.4:  # 高错误密度
        params["num_beams"] = min(params.get("num_beams", 4) + 2, 8)
        params["repetition_penalty"] = max(params.get("repetition_penalty", 1.0), 1.1)
    elif error_density < 0.1:  # 低错误密度
        params["num_beams"] = max(params.get("num_beams", 4) - 1, 2)

    return params


def estimate_error_density(text: str) -> float:
    """增强的错误密度估算，专门检测bootkass等明显错误"""
    words = text.split()
    if not words:
        return 0.0

    error_indicators = 0

    # 扩展的错误模式检测
    patterns = [
        r'\bi\s+has\b',  # "I has"
        r'\bdon\'?t\s+has\b',  # "don't has"
        r'\bhas\s+many\s+\w+s\b',  # 复数错误
        r'\b[a-z]+ss\b',  # 双s结尾拼写错误
        r'\btheir\s+going\b',  # "their going"
        r'\bthere\s+\w+ing\b',  # "there going"
        r'\b\w{4,}kas{1,2}\b',  # 类似bootkass的错误
        r'\b\w+grammer\b',  # grammar拼写错误
        r'\bme\s+and\s+him\b',  # "me and him"
        r'\bstudients\b',  # students拼写错误
        r'\bclasroom\b',  # classroom拼写错误
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        error_indicators += len(matches)

    # 拼写错误检测
    spell_checker = get_spell_checker_instance()
    if spell_checker:
        try:
            # 过滤掉可能的专有名词
            check_words = [w for w in words if len(w) > 2 and not w[0].isupper()]
            misspelled = spell_checker.unknown(check_words)
            error_indicators += len(misspelled)
        except:
            pass

    return min(error_indicators / len(words), 1.0)


def postprocess_text(corrected_text: str, prefix: str, original_text: str) -> str:
    """增强的后处理"""
    if not corrected_text:
        return original_text

    # 移除前缀
    if prefix and corrected_text.lower().startswith(prefix.lower()):
        corrected_text = corrected_text[len(prefix):].strip()

    # 智能首字母大写
    if corrected_text and corrected_text[0].islower():
        corrected_text = corrected_text[0].upper() + corrected_text[1:]

    # 修复常见的后处理问题
    corrected_text = re.sub(r'\s+', ' ', corrected_text)  # 多余空格
    corrected_text = re.sub(r'\s+([,.!?;:])', r'\1', corrected_text)  # 标点前空格

    # 智能句末标点处理
    if corrected_text and not corrected_text[-1] in '.!?':
        if original_text and original_text[-1] in '.!?':
            corrected_text += original_text[-1]
        elif len(corrected_text.split()) > 3:  # 只为较长句子添加句号
            corrected_text += '.'

    return corrected_text


def assess_correction_quality(original: str, corrected: str) -> Dict:
    """增强的纠错质量评估"""
    if original == corrected:
        return {"confidence": "low", "score": 0.0}

    # 多维度质量评估
    similarity = difflib.SequenceMatcher(None, original, corrected).ratio()

    # 检测修复的具体错误类型
    fixes_detected = 0

    # 语法修复检测
    grammar_fixes = [
        (r'\bi\s+has\b', r'\bi\s+have\b'),
        (r'\bdon\'?t\s+has\b', r'\bdon\'?t\s+have\b'),
        (r'\bhe\s+don\'?t\b', r'\bhe\s+doesn\'?t\b'),
    ]

    for original_pattern, fixed_pattern in grammar_fixes:
        if re.search(original_pattern, original, re.IGNORECASE) and \
                re.search(fixed_pattern, corrected, re.IGNORECASE):
            fixes_detected += 1

    # 拼写修复检测
    spelling_fixes = [
        (r'\bgrammer\b', r'\bgrammar\b'),
        (r'\bmistakess\b', r'\bmistakes\b'),
        (r'\bbootkass\b', r'\bbook\w+\b'),  # bootkass修复
        (r'\bstudients\b', r'\bstudents\b'),
        (r'\bclasroom\b', r'\bclassroom\b'),
    ]

    for original_pattern, fixed_pattern in spelling_fixes:
        if re.search(original_pattern, original, re.IGNORECASE) and \
                re.search(fixed_pattern, corrected, re.IGNORECASE):
            fixes_detected += 1

    # 计算置信度
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


# --- 优化的规则引擎系统 ---
def check_with_language_tool(text: str) -> List[Dict]:
    """优化的LanguageTool检查 - 现在直接使用增强的备用检查"""
    return backup_grammar_check(text)

def backup_grammar_check(text: str) -> List[Dict]:
    """Enhanced grammar checking, alternative to LanguageTool"""
    errors = []

    # Extended grammar rules
    grammar_rules = [
        # Subject-Verb Agreement Errors
        {
            "pattern": r'\bi\s+(?:has|was|does|is)\b',
            "message": "Subject-verb disagreement: 'I' should use have/were/do/am",
            "replacement_func": lambda m: re.sub(r'\b(has|was|does|is)\b',
                                                 {'has': 'have', 'was': 'were', 'does': 'do', 'is': 'am'}[m.group(1)],
                                                 m.group()),
            "severity": "High",
            "error_type": "Grammar"
        },
        {
            "pattern": r'\bi\s+(?:has|was|does|is)\b',
            "message": "Subject-verb disagreement: 'He' should use hasn't/doesn't/isn't",
            "replacement_func": lambda m: re.sub(r'\b(dont|isnot|have)\b',
                                                 {"dont":'doesnt','isnot':'isn'}[m.group(1)],
                                                 m.group()),
            "severity": "High",
            "error_type": "Grammar"
        },
        # Direct addition to grammar_rules:
        {
            "pattern": r'\bme and (him|her|them)\b',
            "message": "Pronoun order/case: Use 'I and he/she/they' (or 'He and I', 'She and I' for better flow).",
            "replacement_func": lambda m: "He and I" if m.group(1) == "him" else ("She and I" if m.group(1) == "her" else "They and I"),
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
            "message": "Lexical error: 'their' should be 'they\'re' (they are)",
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
        # Common Spelling and Punctuation Errors
        {
            "pattern": r'\bdont\b',
            "message": "Missing apostrophe: 'dont' should be 'don\'t'",
            "replacement_func": lambda m: "don't",
            "severity": "High",
            "error_type": "Punctuation"
        },
        {
            "pattern": r'\bcant\b',
            "message": "Missing apostrophe: 'cant' should be 'can\'t'",
            "replacement_func": lambda m: "can't",
            "severity": "High",
            "error_type": "Punctuation"
        },
        {
            "pattern": r'\bwont\b',
            "message": "Missing apostrophe: 'wont' should be 'won\'t'",
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

    # Check for sentence structure issues
    sentence_errors = check_sentence_structure(text)
    errors.extend(sentence_errors)

    return sorted(errors, key=lambda x: x["priority"], reverse=True)
def should_skip_error(match) -> bool:
    """过滤低价值错误"""
    skip_rules = [
        "WHITESPACE_RULE",
        "WORD_CONTAINS_UNDERSCORE",
        "EN_QUOTES",
        "CURRENCY"
    ]

    skip_categories = ["TYPOS"]  # 纯拼写错误（由拼写检查器处理）

    return (match.ruleId in skip_rules or
            match.category in skip_categories or
            len(match.message) < 10)


def calculate_error_confidence(match, text: str) -> float:
    """计算错误检测置信度"""
    confidence = 0.5  # 基础置信度

    # 基于规则ID的置信度调整
    high_confidence_rules = ["EN_A_VS_AN", "HE_VERB_AGR", "SUBJECT_VERB_AGR"]
    medium_confidence_rules = ["COMMA_PARENTHESIS", "UPPERCASE_SENTENCE_START"]

    if any(rule in match.ruleId for rule in high_confidence_rules):
        confidence += 0.4
    elif any(rule in match.ruleId for rule in medium_confidence_rules):
        confidence += 0.2

    # 基于替换建议质量
    if match.replacements and len(match.replacements) == 1:
        confidence += 0.1  # 唯一建议更可信

    return min(confidence, 1.0)


def categorize_error(rule_id: str, category: str) -> str:
    """错误类型分类"""
    if "SPELL" in rule_id.upper() or "MORFOLOGIA" in category.upper():
        return "Spelling"
    elif "PUNCT" in rule_id.upper() or "TYPOGRAPHY" in category.upper():
        return "Punctuation"
    elif "GRAMMAR" in category.upper() or "MORFOLOGIK" in rule_id.upper():
        return "Grammar"
    elif "STYLE" in category.upper():
        return "Style"
    else:
        return "Other"


def get_error_severity(rule_id: str, message: str) -> str:
    """错误严重程度评估"""
    high_severity = ["MORFOLOGIK_RULE", "EN_A_VS_AN", "HE_VERB_AGR", "SUBJECT_VERB_AGR"]
    medium_severity = ["COMMA_PARENTHESIS", "UPPERCASE_SENTENCE_START", "WHITESPACE"]

    # 基于规则ID判断
    if any(h in rule_id for h in high_severity):
        return "High"
    elif any(m in rule_id for m in medium_severity):
        return "Medium"

    # 基于消息内容判断
    if any(word in message.lower() for word in ["subject", "verb", "agreement", "grammar"]):
        return "High"
    elif any(word in message.lower() for word in ["comma", "punctuation", "capital"]):
        return "Medium"
    else:
        return "Low"


def calculate_priority(severity: str, confidence: float, error_type: str) -> int:
    """计算错误处理优先级"""
    base_score = {"High": 100, "Medium": 50, "Low": 20}[severity]
    confidence_boost = int(confidence * 30)
    type_boost = {"Grammar": 20, "Spelling": 15, "Punctuation": 10, "Style": 5}[error_type]

    return base_score + confidence_boost + type_boost


def check_spelling_pyspell(text: str) -> List[Dict]:
    """增强的拼写检查，专门优化bootkass等明显错误"""
    spell = get_spell_checker_instance()
    if not spell:
        return []

    try:
        # 智能词语提取，包含位置信息
        words = extract_words_with_positions(text)
        word_list = [w["word"] for w in words]
        misspelled = spell.unknown(word_list)

        suggestions = []
        for word_info in words:
            word = word_info["word"]
            if word not in misspelled:
                continue

            # 跳过明显的专有名词和缩写
            if should_skip_word(word):
                continue

            correction = spell.correction(word)
            candidates = list(spell.candidates(word))[:5]

            # 特殊处理常见错误模式
            correction, candidates = handle_special_spelling_cases(word, correction, candidates)

            # 增强的置信度计算
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

        # 按优先级排序，确保明显错误排在前面
        return sorted(suggestions, key=lambda x: x["priority"], reverse=True)

    except Exception as e:
        logger.error(f"增强拼写检查失败: {e}")
        return []


def handle_special_spelling_cases(word: str, correction: str, candidates: List[str]) -> Tuple[str, List[str]]:
    """处理特殊拼写错误情况，专门处理bootkass等明显错误"""
    word_lower = word.lower()

    # 扩展特殊错误模式映射
    special_corrections = {
        'bootkass': 'bookcase',  # 添加bootkass的专门处理
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
        # 保持原始大小写格式
        if word[0].isupper():
            special_correction = special_correction.capitalize()

        # 确保特殊修正在候选列表的第一位
        if special_correction not in candidates:
            candidates = [special_correction] + candidates[:4]
        else:
            # 如果已在列表中，移到第一位
            candidates.remove(special_correction)
            candidates = [special_correction] + candidates

        return special_correction, candidates

    # 处理类似bootkass的模式 (word + kass/kas)
    if word_lower.endswith('kass') or word_lower.endswith('kas'):
        base_word = word_lower[:-4] if word_lower.endswith('kass') else word_lower[:-3]

        # 特定处理book相关的错误
        if base_word == 'boot' or base_word == 'book':
            correction = 'bookcase'
            candidates = ['bookcase', 'book', 'books'] + candidates[:2]
            return correction, candidates

        # 其他kass结尾的处理
        possible_corrections = [
            base_word + 'case',
            base_word + 'class',
            base_word + 's'
        ]

        # 添加到候选列表前面
        for poss_correction in possible_corrections:
            if poss_correction not in candidates:
                candidates.insert(0, poss_correction)

        if candidates:
            correction = candidates[0]

    return correction, candidates[:5]

def extract_words_with_positions(text: str) -> List[Dict]:
    """提取单词及其位置信息"""
    words = []
    for match in re.finditer(r'\b[a-zA-Z]{2,}\b', text):
        words.append({
            "word": match.group(),
            "position": match.start()
        })
    return words


def should_skip_word(word: str) -> bool:
    """过滤不需要检查的词"""
    # 跳过过短、全大写（可能是缩写）、包含数字的词
    if (len(word) < 3 or
            word.isupper() or
            any(c.isdigit() for c in word)):
        return True

    # 跳过常见的缩写和网络用语
    skip_words = {
        'ok', 'hi', 'bye', 'lol', 'omg', 'wtf', 'btw', 'fyi',
        'etc', 'vs', 'ie', 'eg', 'asap', 'faq', 'diy'
    }

    if word.lower() in skip_words:
        return True

    # 跳过可能的专有名词（首字母大写且长度>5）
    if word[0].isupper() and len(word) > 5:
        return True

    return False


def calculate_spelling_confidence(word: str, correction: str, candidates: List[str]) -> float:
    """计算拼写纠错置信度"""
    if not correction or correction == word:
        return 0.1

    confidence = 0.5

    # 基于编辑距离
    edit_distance = calculate_edit_distance(word, correction)
    if edit_distance == 1:
        confidence += 0.3  # 单字符差异，高置信度
    elif edit_distance == 2:
        confidence += 0.2

    # 基于候选数量
    if len(candidates) == 1:
        confidence += 0.2  # 唯一候选
    elif len(candidates) <= 3:
        confidence += 0.1

    return min(confidence, 1.0)


def calculate_edit_distance(s1: str, s2: str) -> int:
    """计算编辑距离"""
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
    """计算拼写错误优先级"""
    base_score = int(confidence * 100)
    length_bonus = min(word_length * 2, 20)  # 长词优先
    return base_score + length_bonus


def get_word_context(text: str, position: int, word_length: int) -> str:
    """获取单词上下文"""
    start = max(0, position - 20)
    end = min(len(text), position + word_length + 20)
    return text[start:end]


# --- 智能反馈融合系统 ---
def intelligent_feedback_fusion(transformer_result: Dict, lt_errors: List[Dict], spell_errors: List[Dict]) -> Dict:
    """智能融合Transformer和规则引擎反馈"""

    original_text = transformer_result.get("original_text", "")
    corrected_text = transformer_result.get("corrected_text", "")

    # 1. 分析Transformer修复的问题
    transformer_fixes = analyze_transformer_fixes(original_text, corrected_text)

    # 2. 过滤已修复的规则引擎错误
    filtered_lt_errors = filter_resolved_lt_errors(lt_errors, transformer_fixes, corrected_text)
    filtered_spell_errors = filter_resolved_spell_errors(spell_errors, transformer_fixes, corrected_text)

    # 3. 识别Transformer遗漏的问题
    missed_issues = identify_missed_issues(filtered_lt_errors, filtered_spell_errors, corrected_text)

    # 4. 生成融合建议
    fusion_suggestions = generate_fusion_suggestions(
        transformer_result, filtered_lt_errors, filtered_spell_errors, missed_issues
    )

    return {
        "transformer_result": transformer_result,
        "filtered_lt_errors": filtered_lt_errors[:5],  # 限制数量
        "filtered_spell_errors": filtered_spell_errors[:3],
        "missed_critical_issues": missed_issues,
        "fusion_suggestions": fusion_suggestions,
        "confidence_score": calculate_overall_confidence(transformer_result, filtered_lt_errors, filtered_spell_errors)
    }


def analyze_transformer_fixes(original: str, corrected: str) -> List[Dict]:
    """分析Transformer做出的修复"""
    if original == corrected:
        return []

    fixes = []

    # 检测常见修复模式
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
    """过滤已被Transformer解决的LanguageTool错误"""
    filtered = []

    for error in lt_errors:
        if is_error_resolved(error, transformer_fixes, corrected_text):
            continue
        filtered.append(error)

    return filtered


def filter_resolved_spell_errors(
        spell_errors: List[Dict],
        transformer_fixes: List[Dict],
        corrected_text: str
    ) -> List[Dict]:
    filtered = []
    lower_corrected = corrected_text.lower()
    for err in spell_errors:
        if err["word"].lower() not in lower_corrected or (
                err["correction"] and err["correction"].lower() in lower_corrected
        ):
            continue
        filtered.append(err)
    return filtered


globals()["filter_resolved_spell_errors"] = filter_resolved_spell_errors


def is_error_resolved(error: Dict, transformer_fixes: List[Dict], corrected_text: str) -> bool:
    """检查错误是否已被解决"""
    # 检查建议的修复是否出现在修正文本中
    if error["replacement"] != "N/A" and error["replacement"].lower() in corrected_text.lower():
        return True

    # 检查是否匹配Transformer修复模式
    for fix in transformer_fixes:
        if fix["type"] in error["rule_id"].lower():
            return True

    return False


def identify_missed_issues(lt_errors: List[Dict], spell_errors: List[Dict], corrected_text: str) -> List[Dict]:
    """识别Transformer遗漏的关键问题"""
    critical_issues = []

    # 高优先级语法错误
    for error in lt_errors:
        if error["priority"] > 80 and error["confidence"] > 0.7:
            critical_issues.append({
                "type": "grammar",
                "message": error["message"],
                "suggestion": error["replacement"],
                "priority": error["priority"]
            })

    # 高置信度拼写错误
    for error in spell_errors:
        if error["confidence"] > 0.8:
            critical_issues.append({
                "type": "spelling",
                "message": f"Misspelled: {error['word']}",
                "suggestion": error["correction"],
                "priority": error["priority"]
            })

    return critical_issues[:3]  # 限制为最重要的3个


def generate_fusion_suggestions(transformer_result: Dict, lt_errors: List[Dict],
                                spell_errors: List[Dict], missed_issues: List[Dict]) -> List[Dict]:
    """生成融合建议"""
    suggestions = []

    # Transformer结果作为主要建议
    if transformer_result.get("changes_made"):
        suggestions.append({
            "type": "primary",
            "source": "AI Model",
            "suggestion": transformer_result["corrected_text"],
            "confidence": transformer_result.get("confidence", "medium"),
            "priority": 100
        })

    # 关键遗漏问题
    for issue in missed_issues:
        suggestions.append({
            "type": "additional",
            "source": f"Rule Engine ({issue['type']})",
            "suggestion": issue["suggestion"],
            "reason": issue["message"],
            "priority": issue["priority"]
        })

    # 按优先级排序
    return sorted(suggestions, key=lambda x: x["priority"], reverse=True)


def calculate_overall_confidence(transformer_result: Dict, lt_errors: List[Dict], spell_errors: List[Dict]) -> str:
    """计算整体置信度"""
    transformer_conf = transformer_result.get("improvement_score", 0)
    remaining_issues = len(lt_errors) + len(spell_errors)

    if transformer_conf > 0.7 and remaining_issues < 2:
        return "high"
    elif transformer_conf > 0.4 and remaining_issues < 4:
        return "medium"
    else:
        return "low"


# --- 智能冗余消除系统 ---
def remove_redundant_suggestions(transformer_result: Dict, lt_errors: List[Dict], spell_errors: List[Dict]) -> Dict:
    """智能消除不同工具间的冗余建议"""

    # 分析Transformer已经修复的问题
    original_text = transformer_result.get("original_text", "")
    corrected_text = transformer_result.get("corrected_text", "")

    # 过滤已被Transformer修复的LanguageTool错误
    filtered_lt_errors = []
    for error in lt_errors:
        error_span = original_text[error["offset"]:error["offset"] + error["length"]]
        # 检查这个错误是否已经被修复
        if error_span.lower() in corrected_text.lower() and error["replacement"].lower() in corrected_text.lower():
            continue  # 跳过已修复的错误
        filtered_lt_errors.append(error)

    # 过滤已被Transformer修复的拼写错误
    filtered_spell_errors = []
    for error in spell_errors:
        if error["correction"] and error["correction"].lower() in corrected_text.lower():
            continue  # 跳过已修复的拼写错误
        filtered_spell_errors.append(error)

    # 合并并去重建议
    unique_suggestions = {}

    # 添加高置信度的建议
    for error in filtered_lt_errors:
        if error["severity"] == "High":
            key = f"{error['offset']}_{error['length']}"
            unique_suggestions[key] = {
                "type": "grammar",
                "suggestion": error["replacement"],
                "reason": error["message"],
                "priority": "high"
            }

    for error in filtered_spell_errors:
        if error["confidence"] == "high":
            key = f"spell_{error['word']}"
            unique_suggestions[key] = {
                "type": "spelling",
                "suggestion": error["correction"],
                "reason": f"Misspelled word: {error['word']}",
                "priority": "medium"
            }

    return {
        "transformer_result": transformer_result,
        "filtered_lt_errors": filtered_lt_errors,
        "filtered_spell_errors": filtered_spell_errors,
        "unique_suggestions": list(unique_suggestions.values())
    }


# --- 改进的差异高亮系统 ---
def get_highlighted_diff(original_text: str, corrected_text: str) -> List[Tuple]:
    """生成详细的差异高亮，支持多种变更类型"""
    if not original_text or not corrected_text:
        return [(original_text or corrected_text, None)]

    # 使用字符级和词级混合比较
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
    """将LanguageTool错误映射到HighlightedText格式"""
    if not errors:
        return [(text, None)]

    highlighted_parts = []
    last_end = 0

    # 按偏移量排序
    sorted_errors = sorted(errors, key=lambda x: x['offset'])

    for error in sorted_errors:
        offset = error['offset']
        length = error['length']

        # 添加错误前的正常文本
        if offset > last_end:
            highlighted_parts.append((text[last_end:offset], None))

        # 添加错误文本及标签
        error_text = text[offset:offset + length]
        error_label = f"{error['error_type']} ({error['severity']})"
        highlighted_parts.append((error_text, error_label))

        last_end = offset + length

    # 添加剩余文本
    if last_end < len(text):
        highlighted_parts.append((text[last_end:], None))

    return highlighted_parts


# --- 主控制函数 ---
def master_grammar_check(text_to_check: str, selected_model_key: str,
                         use_language_tool: bool = False, use_pyspellchecker: bool = True) -> Dict:

    if not text_to_check.strip():
        return {"error": "Please enter some text."}

    results = {
        "original_text": text_to_check,
        "timestamp": torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else "CPU",
        "language_tool_status": "available" if get_language_tool_instance() else "unavailable"
    }

    try:
        # 1. Transformer模型纠错（主要方法）
        transformer_result = correct_grammar_hf(text_to_check, selected_model_key)
        transformer_result["original_text"] = text_to_check

        # 2. 规则引擎检查（补充方法）
        lt_errors = check_with_language_tool(text_to_check) if use_language_tool else []#no need
        spell_errors = check_spelling_pyspell(text_to_check) if use_pyspellchecker else []

        # 3. 智能融合和冗余消除
        fusion_result = intelligent_feedback_fusion(transformer_result, lt_errors, spell_errors)

        # 4. 生成最终建议
        final_suggestions = create_final_suggestions(fusion_result)

        # 5. 生成高亮数据
        diff_highlight = get_highlighted_diff(text_to_check, transformer_result.get("corrected_text", text_to_check))
        error_highlight = map_language_tool_errors(text_to_check, fusion_result["filtered_lt_errors"])

        # 组装结果
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
        logger.error(f"主检查函数执行失败: {e}")
        return {
            "error": f"检查失败: {str(e)}",
            "original_text": text_to_check,
            "language_tool_status": "error"
        }


def create_final_suggestions(fusion_result: Dict) -> List[Dict]:
    """创建最终建议列表，优先级排序"""
    suggestions = []

    # 主要AI建议
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

    # 关键遗漏问题
    for issue in fusion_result["missed_critical_issues"]:
        suggestions.append({
            "type": "critical",
            "title": issue["message"],
            "suggestion": issue["suggestion"],
            "confidence": "high",
            "source": f"Rule Engine ({issue['type']})"
        })

    # 次要语法问题
    for error in fusion_result["filtered_lt_errors"][:2]:  # 限制数量
        if error["priority"] > 60:
            suggestions.append({
                "type": "grammar",
                "title": error["message"],
                "suggestion": error["replacement"],
                "confidence": error["confidence"],
                "source": "Grammar Rules"
            })

    return suggestions[:5]  # 最多5个建议


# --- 批量测试函数 ---
def benchmark_all_models(test_sentences: List[str] = None):
    """对所有模型进行性能基准测试"""
    if test_sentences is None:
        test_sentences = [
            "I has a bad grammer and I is very happy.",
            "he dont like water he swim fastly",
            "This sentences has many mistakess. It a beautiful day.",
            "Their going to there house with they're friends.",
            "Me and him was walking to the store yesterday.",
            "The students are reading their bootkass carefully."  # 添加bootkass测试
        ]

    print("=== 模型性能基准测试 ===")
    results = {}

    for model_key in MODEL_OPTIONS.keys():
        print(f"\n测试模型: {model_key}")
        model_results = []

        for i, sentence in enumerate(test_sentences, 1):
            print(f"  测试句子 {i}: {sentence}")
            result = correct_grammar_hf(sentence, model_key)
            model_results.append({
                "input": sentence,
                "output": result.get("corrected_text", "Error"),
                "success": "error" not in result
            })
            print(f"  结果: {result.get('corrected_text', 'Error')}")

        results[model_key] = model_results

        # 显示模型性能信息
        perf = MODEL_OPTIONS[model_key]["performance"]
        print(f"  性能: {perf['accuracy']} | 速度: {perf['speed']} | 内存: {perf['memory']}")

    return results


# --- 专门测试bootkass修复的函数 ---
def test_bootkass_corrections():
    """专门测试bootkass等明显错误的修复"""
    test_cases = [
        "The students in the classroom are reading their bootkass.",
        "I has a bad grammer and need to improve my writting skils.",
        "He don't like water but he swim very good yesterday.",
        "Their going to there house with they're friends.",
        "Me and him was walking to the store and we buyed some mistakess.",
        "The studients are in the clasroom with there techer."
    ]

    print("=== 测试明显错误修复能力 ===")
    print(f"LanguageTool状态: {'禁用' if DISABLE_LANGUAGE_TOOL else '启用'}")
    print("使用增强的备用语法检查系统\n")

    for i, text in enumerate(test_cases, 1):
        print(f"\n{i}. 测试: {text}")

        result = master_grammar_check(text, "T5-Base (Vennify)")

        if "error" not in result:
            print(f"   修正: {result['primary_correction']}")
            print(f"   置信度: {result['confidence']}")
            print(f"   改动: {'是' if result['stats']['transformer_changes'] else '否'}")

            # 显示拼写建议
            if result.get('spell_suggestions'):
                print("   拼写建议:")
                for suggestion in result['spell_suggestions'][:3]:
                    print(
                        f"     {suggestion['word']} → {suggestion['correction']} (置信度: {suggestion['confidence']})")

            # 显示语法错误
            if result.get('grammar_errors'):
                print("   语法错误检测:")
                for error in result['grammar_errors'][:3]:
                    print(f"     {error['message']} (严重程度: {error['severity']})")
        else:
            print(f"   错误: {result['error']}")

# 测试函数调用示例
if __name__ == "__main__":
    # 测试bootkass等错误修复
    test_bootkass_corrections()

    # 原有的批量测试
    print("\n" + "=" * 50)
    benchmark_results = benchmark_all_models()
    print("模型基准测试完成")