"""
大模型养护建议生成模块
支持两种模式：
1. Ollama 本地部署 (推荐) - 使用 Qwen2.5 等本地模型
2. Transformers 本地推理 - 使用轻量对话模型
"""
import os
import sys

from llm.prompts import (
    CARE_SYSTEM_PROMPT, CARE_ADVICE_TEMPLATE,
    BREED_COMPARE_TEMPLATE, HEALTH_QUERY_TEMPLATE, TRAINING_TEMPLATE
)


class PetCareAdvisor:
    """宠物养护智能顾问"""

    def __init__(self, backend="ollama", model_name="qwen2.5:1.5b"):
        """
        :param backend: "ollama" 或 "transformers"
        :param model_name: 模型名称
        """
        self.backend = backend
        self.model_name = model_name
        self.model = None
        self.tokenizer = None

        if backend == "ollama":
            self._check_ollama()
        elif backend == "transformers":
            self._load_transformers_model()
        elif backend == "fallback":
            print("使用离线规则模式")
        else:
            raise ValueError(f"不支持的 backend: {backend}")

    def _check_ollama(self):
        """检查 Ollama 是否可用"""
        try:
            import ollama
            self.ollama = ollama
            # 检查模型是否存在 - 兼容不同版本的返回格式
            result = ollama.list()
            if isinstance(result, dict):
                model_list = result.get("models", [])
            elif isinstance(result, list):
                model_list = result
            else:
                model_list = []
            model_names = []
            for m in model_list:
                if isinstance(m, dict):
                    full_name = m.get("name", m.get("model", ""))
                else:
                    full_name = str(m)
                model_names.append(full_name.split(":")[0])
            base_name = self.model_name.split(":")[0]
            if base_name not in model_names:
                print(f"Ollama 模型 '{self.model_name}' 未找到，正在拉取...")
                ollama.pull(self.model_name)
                print(f"模型 '{self.model_name}' 拉取完成")
        except Exception as e:
            print(f"Ollama 连接失败: {e}")
            print("自动降级为离线模式，可切换到页面侧边栏选择 fallback")
            self.backend = "fallback"

    def _load_transformers_model(self):
        """加载本地 Transformers 模型（轻量量化版本）"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            print(f"正在加载模型: {self.model_name} ...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype="auto"
            )
            print("模型加载完成!")
        except Exception as e:
            print(f"模型加载失败: {e}")
            print("自动降级为离线模式")
            self.backend = "fallback"

    def ask(self, prompt, system_prompt=CARE_SYSTEM_PROMPT):
        """调用 LLM 生成回复"""
        if self.backend == "ollama":
            return self._ask_ollama(prompt, system_prompt)
        elif self.backend == "transformers":
            return self._ask_transformers(prompt, system_prompt)
        else:
            return self._ask_fallback(prompt)

    def _ask_ollama(self, prompt, system_prompt):
        """通过 Ollama API 调用"""
        response = self.ollama.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            options={"temperature": 0.7, "top_p": 0.9}
        )
        return response["message"]["content"]

    def _ask_transformers(self, prompt, system_prompt):
        """通过 Transformers 本地推理"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        response = self.tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True
        )
        return response

    def _ask_fallback(self, prompt):
        """离线后备模式 - 基于品种名返回预设建议"""
        return self._generate_care_advice_fallback(prompt)

    # ========== 流式输出接口 ==========

    def ask_stream(self, prompt, system_prompt=CARE_SYSTEM_PROMPT):
        """流式调用 LLM 生成回复，返回生成器"""
        if self.backend == "ollama":
            yield from self._ask_ollama_stream(prompt, system_prompt)
        elif self.backend == "transformers":
            yield from self._ask_transformers_stream(prompt, system_prompt)
        else:
            yield from self._ask_fallback_stream(prompt)

    def _ask_ollama_stream(self, prompt, system_prompt):
        """Ollama 流式调用"""
        stream = self.ollama.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            stream=True,
            options={"temperature": 0.7, "top_p": 0.9}
        )
        for chunk in stream:
            content = chunk.get("message", {}).get("content", "")
            if content:
                yield content

    def _ask_transformers_stream(self, prompt, system_prompt):
        """Transformers 流式推理"""
        from transformers import TextStreamer
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # 用 TextStreamer 实现流式输出
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        import threading
        from queue import Queue

        q = Queue()
        def generate():
            self.model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                streamer=streamer,
            )
            q.put(None)  # 结束信号

        thread = threading.Thread(target=generate)
        thread.start()
        thread.join()
        yield ""  # TextStreamer 直接输出到 stdout，这里返回空

    def _ask_fallback_stream(self, prompt):
        """离线模式流式输出（逐字符产出模拟流式体验）"""
        text = self._generate_care_advice_fallback(prompt)
        chunk_size = 3
        for i in range(0, len(text), chunk_size):
            import time
            time.sleep(0.02)
            yield text[i:i+chunk_size]

    def get_care_advice_stream(self, breed, breed_cn, pet_type="猫/犬"):
        """流式获取养护建议"""
        prompt = CARE_ADVICE_TEMPLATE.format(
            breed=breed, breed_cn=breed_cn, pet_type=pet_type
        )
        try:
            yield from self.ask_stream(prompt)
        except Exception as e:
            print(f"LLM流式调用失败: {e}")
            yield from self._ask_fallback_stream(prompt)

    def compare_breeds_stream(self, breed_a, breed_a_cn, breed_b, breed_b_cn):
        """流式对比品种"""
        prompt = BREED_COMPARE_TEMPLATE.format(
            breed_a=breed_a, breed_a_cn=breed_a_cn,
            breed_b=breed_b, breed_b_cn=breed_b_cn
        )
        yield from self.ask_stream(prompt)

    def health_advice_stream(self, breed_cn, question):
        """流式健康咨询"""
        prompt = HEALTH_QUERY_TEMPLATE.format(
            breed_cn=breed_cn, user_question=question
        )
        yield from self.ask_stream(prompt)

    def training_advice_stream(self, breed, breed_cn, goal):
        """流式训练建议"""
        prompt = TRAINING_TEMPLATE.format(
            breed=breed, breed_cn=breed_cn, training_goal=goal
        )
        yield from self.ask_stream(prompt)

    # ========== 非流式接口（保留兼容） ==========

    def get_care_advice(self, breed, breed_cn, pet_type="猫/犬"):
        """获取宠物养护建议"""
        prompt = CARE_ADVICE_TEMPLATE.format(
            breed=breed, breed_cn=breed_cn, pet_type=pet_type
        )
        try:
            return self.ask(prompt)
        except Exception as e:
            print(f"LLM调用失败，使用后备方案: {e}")
            return self._generate_care_advice_fallback(prompt)

    def compare_breeds(self, breed_a, breed_a_cn, breed_b, breed_b_cn):
        """对比两个品种"""
        prompt = BREED_COMPARE_TEMPLATE.format(
            breed_a=breed_a, breed_a_cn=breed_a_cn,
            breed_b=breed_b, breed_b_cn=breed_b_cn
        )
        return self.ask(prompt)

    def health_advice(self, breed_cn, question):
        """健康问题咨询"""
        prompt = HEALTH_QUERY_TEMPLATE.format(
            breed_cn=breed_cn, user_question=question
        )
        return self.ask(prompt)

    def training_advice(self, breed, breed_cn, goal):
        """训练建议"""
        prompt = TRAINING_TEMPLATE.format(
            breed=breed, breed_cn=breed_cn, training_goal=goal
        )
        return self.ask(prompt)

    def _generate_care_advice_fallback(self, prompt_text):
        """离线后备：根据品种类型生成通用养护建议"""
        import re
        breed_match = re.search(r'宠物品种[：:]\s*(.+?)[（(]', prompt_text)
        breed_name = breed_match.group(1).strip() if breed_match else "您的宠物"

        is_cat = any(kw in prompt_text for kw in ["猫", "Cat"])
        is_dog = any(kw in prompt_text for kw in ["犬", "Dog"])

        advice = f"""【{breed_name} 养护建议】（AI离线模式通用建议）

一、饮食建议
"""
        if is_cat:
            advice += """• 选择优质猫粮，确保蛋白质含量≥30%
• 每日喂食2-3次，成猫每日约50-70克干粮
• 提供充足的清洁饮水，建议使用流动饮水机
• 禁忌食物：洋葱、大蒜、巧克力、葡萄、牛奶（乳糖不耐受）
"""
        elif is_dog:
            advice += """• 选择适合体型的优质狗粮，注意蛋白质与脂肪配比
• 每日喂食2次，幼犬3-4次，定时定量
• 配备充足的清洁饮用水
• 禁忌食物：巧克力、洋葱、葡萄、木糖醇、生肉
"""
        else:
            advice += """• 选择正规品牌的宠物粮，注意营养均衡
• 定时定量喂食，避免暴饮暴食
• 提供充足的清洁饮水
• 避免喂食人类调味食品
"""

        advice += """
二、日常护理
• 定期梳理毛发，短毛品种每周2-3次，长毛品种每日梳理
• 洗澡频率：每月1-2次（夏季），2-3月1次（冬季），使用宠物专用香波
• 每周检查并清洁耳朵，每月修剪指甲
• 定期进行体内外驱虫（每3个月一次）

三、健康管理
• 幼宠完成基础疫苗接种（猫三联/犬五联等），成年后每年加强一次
• 每年至少一次全面体检，老年宠物建议每半年一次
• 留意异常症状：食欲不振、精神萎靡、呕吐腹泻、异常脱毛等
• 建立宠物健康档案，记录疫苗接种和就医信息

四、运动与训练
• 每日保证适量运动，犬类需外出散步30-60分钟，猫类提供玩具互动
• 正向激励训练法：用零食和表扬奖励正确行为
• 建立固定作息，培养良好生活习惯
• 循序渐进训练，每次10-15分钟，保持耐心

五、生活环境
• 保持居住环境清洁通风，冬暖夏凉
• 犬类准备舒适的狗窝/笼子，猫类提供猫抓板和高处活动空间
• 避免宠物接触有毒植物和危险物品
• 给予足够的陪伴与关爱，关注宠物心理健康

⚠️ 提示：以上为通用养护建议。已连接大模型时可获取针对该品种的专业详细建议。
"""
        return advice
