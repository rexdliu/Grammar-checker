# -*- coding: utf-8 -*-
"""
高级英语语法检查系统 - 深度优化版本
支持GPU强制运行，多模型深度集成，智能冗余消除
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration
import torch
import language_tool_python
from spellchecker import SpellChecker
import difflib
import re
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import warnings

warnings.filterwarnings("ignore")


# --- GPU设备配置 ---
def setup_device():
    if torch.cuda.is_available():
        device = "cuda"
        torch.cuda.empty_cache()
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return device
    else:
        print("Using CPU")
        return "cpu"
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

    "Coedit-Large (Grammarly)": {
        "name": "grammarly/coedit-large",
        "prefix": "Fix the grammar: ",  # 修复前缀
        "model_type": "T5ForConditionalGeneration",
        "params": {
            "max_length": 512,
            "num_beams": 4,
            "early_stopping": True,
            "no_repeat_ngram_size": 2,
            "repetition_penalty": 1.0,  # 降低惩罚
            "length_penalty": 1.0,
            "do_sample": True,
            "temperature": 0.3
        },
        "performance": {
            "accuracy": "Very High",
            "speed": "Medium",
            "memory": "~6GB",
            "strengths": "复杂语法错误，指令理解，语义保持",
            "weaknesses": "资源消耗大，推理较慢"
        }
    },

    "FLAN-T5-Large (Pszemraj)": {
        "name": "pszemraj/flan-t5-large-grammar-synthesis",
        "prefix": "grammar: ",  # 只保留这一行
        "model_type": "AutoModelForSeq2SeqLM",
        "params": {
            "max_length": 512,
            "num_beams": 4,
            "early_stopping": True,
            "no_repeat_ngram_size": 2,
            "repetition_penalty": 1.0,
            "length_penalty": 0.8,
            "do_sample": True,  # 改为True
            "temperature": 0.7  # 添加这行
        },
        "performance": {
            "accuracy": "Very High",
            "speed": "Slow",
            "memory": "~5GB",
            "strengths": "单次全文纠错，语义完整性，无需前缀",
            "weaknesses": "大文本处理时间长"
        }
    },

    "Error Corrector v1 (Prithivida)": {
        "name": "prithivida/grammar_error_correcter_v1",
        "prefix": "",  # Gramformer无需前缀
        "model_type": "AutoModelForSeq2SeqLM",
        "params": {
            "max_length": 128,  # 轻量化设置
            "num_beams": 3,  # 降低beam数提升速度
            "early_stopping": True,
            "no_repeat_ngram_size": 2,
            "repetition_penalty": 1.0,
            "length_penalty": 1.0,
            "do_sample": False
        },
        "performance": {
            "accuracy": "Medium",
            "speed": "Very Fast",
            "memory": "~1GB",
            "strengths": "轻量化，快速处理，低资源消耗",
            "weaknesses": "复杂错误处理能力有限，较老模型"
        }
    }
}

# --- 全局实例管理 ---
_language_tool_instance = None
_spell_checker_instance = None
loaded_models = {}


def get_language_tool_instance():
    """获取LanguageTool单例"""
    global _language_tool_instance
    if _language_tool_instance is None:
        print("Initializing LanguageTool...")
        try:
            _language_tool_instance = language_tool_python.LanguageTool('en-US')
            print("LanguageTool initialized successfully.")
        except Exception as e:
            print(f"LanguageTool initialization failed: {e}")
            _language_tool_instance = None
    return _language_tool_instance


def get_spell_checker_instance():
    """获取SpellChecker单例"""
    global _spell_checker_instance
    if _spell_checker_instance is None:
        print("Initializing SpellChecker...")
        _spell_checker_instance = SpellChecker()
        print("SpellChecker initialized.")
    return _spell_checker_instance


#
def load_hf_model(model_name: str, model_type: str):
    """加载并缓存Hugging Face模型，支持不同模型类型"""
    if model_name not in loaded_models:
        print(f"Loading model: {model_name} (Type: {model_type})")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            # 确保有pad_token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            # 根据模型类型选择不同的加载方式
            if model_type == "T5ForConditionalGeneration":
                model = T5ForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,  # 使用半精度节省内存
                    device_map="auto"
                )
            else:  # AutoModelForSeq2SeqLM
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto" if "large" in model_name.lower() else None
                ).to(DEVICE)

            model.eval()
            loaded_models[model_name] = {"tokenizer": tokenizer, "model": model}
            print(f"Successfully loaded {model_name}")

        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            return None, None

    return loaded_models[model_name]["tokenizer"], loaded_models[model_name]["model"]


# --- core function---
def correct_grammar_hf(text: str, model_key: str) -> Dict:
    """使用指定模型进行语法纠错，返回详细结果"""
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

    # preprocess
    cleaned_text = preprocess_text(text)
    input_text = f"{prefix}{cleaned_text}" if prefix else cleaned_text

    try:
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
            add_special_tokens=True
        ).to(DEVICE)
        with torch.no_grad():
            input_length = inputs.input_ids.shape[1]

            # 动态参数调整
            dynamic_params = adjust_generation_params(params, input_length, text)

            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=dynamic_params["max_length"],
                min_length=dynamic_params.get("min_length", 1),
                num_beams=dynamic_params["num_beams"],
                early_stopping=dynamic_params["early_stopping"],
                no_repeat_ngram_size=dynamic_params.get("no_repeat_ngram_size", 2),
                repetition_penalty=dynamic_params["repetition_penalty"],
                length_penalty=dynamic_params["length_penalty"],
                do_sample=dynamic_params["do_sample"],
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 高级后处理
        corrected_text = postprocess_text(corrected_text, prefix, text)

        # 质量评估
        quality_score = assess_correction_quality(text, corrected_text)

        return {
            "corrected_text": corrected_text,
            "model_used": model_key,
            "confidence": quality_score["confidence"],
            "changes_made": corrected_text != text,
            "improvement_score": quality_score["score"],
            "processing_time": "optimized"
        }

    except Exception as e:
        return {"error": f"Generation failed: {str(e)}", "corrected_text": text}


def preprocess_text(text: str) -> str:
    """文本预处理：标准化格式"""
    # 移除多余空格
    text = re.sub(r'\s+', ' ', text.strip())

    # 修复明显的格式问题
    text = re.sub(r'\s+([,.!?;:])', r'\1', text)  # 标点前空格
    text = re.sub(r'([.!?])\s*([a-z])', r'\1 \2', text)  # 句子间距

    return text


def adjust_generation_params(base_params: Dict, input_length: int, text: str) -> Dict:
    """动态调整生成参数"""
    params = base_params.copy()

    # 根据输入长度调整max_length
    if input_length > 100:
        params["max_length"] = min(512, int(input_length * 1.5))
    else:
        params["max_length"] = min(256, max(50, int(input_length * 2.0)))

    # 根据文本复杂度调整beam数
    error_density = estimate_error_density(text)
    if error_density > 0.3:  # 高错误密度
        params["num_beams"] = min(params["num_beams"] + 2, 8)

    return params


def estimate_error_density(text: str) -> float:
    """估算文本错误密度"""
    words = text.split()
    if not words:
        return 0.0

    # 简单的错误指标
    error_indicators = 0

    # 检查常见错误模式
    patterns = [
        r'\bi\s+has\b',  # "I has"
        r'\bdon\'?t\s+like\b',  # "dont like"
        r'\bhas\s+many\s+\w+ss\b',  # 复数错误
        r'\b[a-z]+ss\b'  # 双s结尾可能的拼写错误
    ]

    for pattern in patterns:
        error_indicators += len(re.findall(pattern, text, re.IGNORECASE))

    return min(error_indicators / len(words), 1.0)


def postprocess_text(corrected_text: str, prefix: str, original_text: str) -> str:
    """移除前缀，格式优化"""
    # 移除前缀
    if prefix and corrected_text.lower().startswith(prefix.lower()):
        corrected_text = corrected_text[len(prefix):].strip()

    if corrected_text and corrected_text[0].islower():
        corrected_text = corrected_text[0].upper() + corrected_text[1:]
    # 确保句末标点
    if corrected_text and not corrected_text[-1] in '.!?':
        if original_text and original_text[-1] in '.!?':
            corrected_text += original_text[-1]
        else:
            corrected_text += '.'

    return corrected_text


def assess_correction_quality(original: str, corrected: str) -> Dict:

    if original == corrected:
        return {"confidence": "low", "score": 0.0}

    # 计算改进指标
    original_errors = len(re.findall(r'\b(has|is)\s+\w+\b', original.lower()))
    corrected_errors = len(re.findall(r'\b(has|is)\s+\w+\b', corrected.lower()))

    improvement = max(0, original_errors - corrected_errors)
    score = min(improvement / max(original_errors, 1), 1.0)

    if score > 0.7:
        confidence = "high"
    elif score > 0.3:
        confidence = "medium"
    else:
        confidence = "low"

    return {"confidence": confidence, "score": score}


# --- 优化的规则引擎系统 ---
def check_with_language_tool(text: str) -> List[Dict]:
    """优化的LanguageTool检查，智能过滤和分类"""
    tool = get_language_tool_instance()
    if not tool:
        return []

    try:
        matches = tool.check(text)
        errors = []

        for match in matches:
            # 过滤低价值错误
            if should_skip_error(match):
                continue

            error_type = categorize_error(match.ruleId, match.category)
            severity = get_error_severity(match.ruleId, match.message)
            confidence = calculate_error_confidence(match, text)

            errors.append({
                "message": match.message,
                "replacement": match.replacements[0] if match.replacements else "N/A",
                "offset": match.offset,
                "length": match.errorLength,
                "context": match.context,
                "rule_id": match.ruleId,
                "category": match.category,
                "error_type": error_type,
                "severity": severity,
                "confidence": confidence,
                "priority": calculate_priority(severity, confidence, error_type)
            })

        # 按优先级排序
        return sorted(errors, key=lambda x: x["priority"], reverse=True)

    except Exception as e:
        print(f"LanguageTool error: {e}")
        return []


def should_skip_error(match) -> bool:
    """过滤低价值错误"""
    skip_rules = [
        "WHITESPACE_RULE",  # 空格问题
        "WORD_CONTAINS_UNDERSCORE",  # 下划线
        "EN_QUOTES",  # 引号样式
        "CURRENCY"  # 货币格式
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
    """增强的拼写检查，智能过滤和置信度评估"""
    spell = get_spell_checker_instance()
    if not spell:
        return []

    try:
        # 智能词语提取
        words = extract_words_with_positions(text)
        word_list = [w["word"] for w in words]
        misspelled = spell.unknown(word_list)

        suggestions = []
        for word_info in words:
            word = word_info["word"]
            if word not in misspelled:
                continue

            # 过滤明显的非单词
            if should_skip_word(word):
                continue

            correction = spell.correction(word)
            candidates = list(spell.candidates(word))[:3]

            # 计算拼写错误置信度
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

        # 按优先级排序
        return sorted(suggestions, key=lambda x: x["priority"], reverse=True)

    except Exception as e:
        print(f"SpellChecker error: {e}")
        return []


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
    return (len(word) < 3 or
            word.isupper() or
            any(c.isdigit() for c in word) or
            word.lower() in ["ok", "hi", "bye"])


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
        (r'\bmistakess\b', r'\bmistakes\b', "spelling_fix")
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


def filter_resolved_spell_errors(spell_errors: List[Dict], transformer_fixes: List[Dict], corrected_text: str) -> List[
    Dict]:
    """过滤已被Transformer解决的拼写错误"""
    filtered = []

    for error in spell_errors:
        if error["correction"] and error["correction"].lower() in corrected_text.lower():
            continue
        filtered.append(error)

    return filtered


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


# --- 更新主控制函数 ---
def master_grammar_check(text_to_check: str, selected_model_key: str,
                         use_language_tool: bool = True, use_pyspellchecker: bool = True) -> Dict:
    """智能语法检查主函数，融合多种工具并消除冗余"""

    if not text_to_check.strip():
        return {"error": "Please enter some text."}

    results = {
        "original_text": text_to_check,
        "timestamp": torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else "CPU"
    }

    # 1. Transformer模型纠错（主要方法）
    transformer_result = correct_grammar_hf(text_to_check, selected_model_key)
    transformer_result["original_text"] = text_to_check

    # 2. 规则引擎检查（补充方法）
    lt_errors = check_with_language_tool(text_to_check) if use_language_tool else []
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
        "stats": {
            "transformer_changes": transformer_result.get("changes_made", False),
            "grammar_issues_found": len(fusion_result["filtered_lt_errors"]),
            "spelling_issues_found": len(fusion_result["filtered_spell_errors"]),
            "redundancy_reduction": f"{len(lt_errors + spell_errors) - len(fusion_result['filtered_lt_errors'] + fusion_result['filtered_spell_errors'])} duplicates removed"
        }
    })

    return results


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
            "Me and him was walking to the store yesterday."
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


# 测试函数调用示例
if __name__ == "__main__":
    # 测试单个纠错
    test_text = "I has a bad grammer and I is very happy."
   # result = master_grammar_check(test_text, "T5-Base (Vennify)")


    # 批量测试所有模型
    benchmark_results = benchmark_all_models()
    print("测试结果:", benchmark_results )
