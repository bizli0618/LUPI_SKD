import json
import random
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase
from typing import List, Dict, Tuple

class LUPIDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: PreTrainedTokenizerBase, max_length: int = 4096):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            
        print(f"[LUPIDataset] Loaded {len(self.data)} samples from {data_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        query = item['query']
        knowledge = item['knowledge']
        
        # Reuse the build_lupi_inputs logic from the notebook
        teacher_inputs, student_inputs = self._build_inputs(query, knowledge)
        
        return {
            "teacher_input_ids": teacher_inputs['input_ids'].squeeze(0),
            "student_input_ids": student_inputs['input_ids'].squeeze(0),
            "teacher_attention_mask": teacher_inputs['attention_mask'].squeeze(0),
            "student_attention_mask": student_inputs['attention_mask'].squeeze(0)
        }

    def _build_inputs(self, query: str, knowledge: str):
        # 1. System Prompt (English Version)
        system_prompt = (
            "You are an expert on Korean history and culture. "
            "Answer the user's question based on the provided [Background Knowledge].\n"
            "1. Infer and reason based on the facts (years, figures, events, etc.) described in [Background Knowledge].\n"
            "2. Write the answer naturally as if it is your own knowledge, avoiding direct citation formats.\n"
            "3. Do NOT include information not present in [Background Knowledge] unless you are absolutely certain of its factual accuracy."
        )
        
        # 2. Few-Shot Examples (All English: Q -> CoT + A)
        
        # Shot 1: Food/Fermentation
        shot1_k = (
            "Sunlight plays a crucial role in making soy sauce. When meju (fermented soybean lump) is soaked in salt water and placed in a jar in a sunny spot, "
            "the ultraviolet rays from the sun suppress harmful bacteria and allow only beneficial microbes to survive. "
            "Specifically, salt-tolerant yeasts and certain lactic acid bacteria thrive, creating a savory and deep flavor. "
            "Temperature changes caused by sunlight are also important; the warmth during the day accelerates fermentation, while the coolness at night allows for slow maturation, "
            "deepening the taste. Additionally, the photo-oxidation reaction from sunlight creates a substance called melanoidin, "
            "which gives soy sauce its dark color and savory aroma."
        )
        shot1_q = "Why do you place the jar in the sun after soaking meju in salt water when making soy sauce?"
        shot1_a = (
            "<think>\n"
            "Okay, so the user is asking why you put the meju in salt water and then leave it in the sun when making soy sauce. Let me break this down.\n\n"
            "First, you soak it in salt water, which is high in salinity. This probably helps kill off unwanted microbes. But why then expose it to sunlight?\n"
            "Hmm, sunlight has UV rays. Those UV rays might act as a natural disinfectant, suppressing harmful bacteria. On the other hand, maybe beneficial microbes that tolerate salt thrive here. So the sun helps select which microorganisms grow.\n\n"
            "Also, temperature changes from the sun must play a role. During the day, the temperature rises, speeding up fermentation. At night, it gets cooler, slowing it down. This cycle might help the flavor mature deeply.\n\n"
            "Another thing is the chemical reaction. Exposure to sunlight might cause photo-oxidation. I recall reading about melanoidins forming, which give soy sauce its dark color and rich taste.\n\n"
            "So, putting it all together: sunlight helps kill bad microbes, encourages good salt-tolerant ones, controls fermentation through temperature swings, and develops color and flavor. It's essential for making good soy sauce.\n"
            "</think>\n\n"
            "The reasons for placing the jar in the sun after soaking meju in salt water when making soy sauce are as follows, providing significant benefits:\n\n"
            "1. **Selective Growth of Microbes**: When the ultraviolet rays from sunlight are added to the saline environment, harmful bacteria are suppressed, and beneficial salt-tolerant yeasts and lactic acid bacteria survive, creating a savory flavor.\n"
            "2. **Regulation of Fermentation and Maturation**: During the day, the warmth from the sunlight accelerates fermentation, while at night, the cooler temperatures allow for slow maturation. This temperature fluctuation deepens the flavor of the sauce.\n"
            "3. **Completion of Color and Aroma**: A photochemical reaction caused by sunlight creates a substance called 'melanoidin,' which produces the unique dark brown color and deep aroma of soy sauce.\n\n"
            "Therefore, placing the jangdokdae (jar stand) in a sunny spot is traditional wisdom to maximize these sterilization and fermentation effects."
        )

        # Shot 2: History
        shot2_k = (
            "King Gwanggaeto (reigned 391-412) was the 19th monarch of Goguryeo. During his 21-year reign, he conquered 64 fortresses and over 1,400 villages, "
            "developing Goguryeo into a great empire. His major conquests include the following: In July 392, he attacked Baekje and captured over 10 fortresses including Seokhyeonseong. "
            "In 396, he captured 58 fortresses and 700 villages, securing the region north of the Han River. "
            "In 395, he defeated the three tribes of Khitan (Bili), and in 398, he conquered Sushen, stabilizing the northeastern border. "
            "In 400, he responded to Silla's request for aid by dispatching 50,000 troops to repel the Japanese invaders (Wa). "
            "In the war against Later Yan, he captured Sukgunseong in 402, Yodongseong in 405, and Mokjeoseong in 406. In 407, he mobilized 50,000 troops to capture 6 fortresses. "
            "In 410, he conquered Dongbuyeo."
        )
        shot2_q = "What were the major battles and the scale of territory expansion during King Gwanggaeto's reign?"
        shot2_a = (
            "<think>\n"
            "Okay, I need to list the main battles and territory expansion of King Gwanggaeto.\n"
            "The user mentioned he ruled for 21 years, conquering 64 cities and 1,400 villages.\n"
            "Key battles:\n"
            "In 392 and 396, he attacked Baekje, taking many cities and securing the Han River area.\n"
            "In the north, he defeated Khitan (395) and Sushin (398).\n"
            "In the west, he fought Later Yan (402-407), taking Liaodong fortress.\n"
            "In 400, he sent 50,000 troops to Silla to fight the Wa (Japan).\n"
            "Finally, he conquered Dongbuyeo in 410.\n"
            "The territory reached from Songhua River to Han River, and Liaodong to the East Sea.\n"
            "I will explain this in detail, highlighting the scale and specific regions.\n"
            "</think>\n\n"
            "King Gwanggaeto developed Goguryeo into the strongest power in Northeast Asia by conquering 64 fortresses and over 1,400 villages during his 21-year reign. The major battles and territorial expansion can be summarized as follows:\n\n"
            "First, to expand the southern border, he attacked Baekje in 392, capturing about 10 fortresses, and notably in 396, he captured 58 fortresses and 700 villages, seizing control of the region north of the Han River. "
            "In the northeast, he defeated the Khitan in 395 and conquered Sushen in 398, stabilizing the border, which extended the northern territory to the Songhua River basin.\n\n"
            "To the west, through continuous wars with Later Yan, he successively captured Sukgunseong in 402 and Yodongseong in 405, and in 407, he mobilized 50,000 troops to completely secure the Liaodong region. "
            "Also, in 400, he responded to Silla's request for aid by dispatching 50,000 troops to repel the Japanese invaders (Wa), exerting influence over the southern part of the Korean Peninsula as well.\n\n"
            "As a result, Goguryeo came to occupy a vast territory extending from the Songhua River in the north to the Han River in the south, and from Liaodong in the west to the East Sea coast in the east. "
            "These conquests served as the decisive foundation for Goguryeo's golden age."
        )

        # Shot 3: Tradition
        shot3_k = (
            "Charye (Tea Ritual) and Jesa (Ancestral Rite) are both ceremonies to honor ancestors, but their purpose and procedures differ. "
            "Charye is a simple rite performed on holidays (Seollal, Chuseok, etc.), usually held in the morning (daytime), and tea or sikhye is often offered instead of alcohol. "
            "The food consists mainly of holiday dishes (tteokguk, songpyeon, etc.) and is set simply, and the procedure is relatively simple. "
            "On the other hand, Gijesa (Memorial Rite) is a formal rite held on the anniversary of the ancestor's death. It starts right after midnight or in the early morning (0-2 AM), "
            "and follows complex procedures such as Heonjak (offering alcohol). The sacrificial table is also set according to strict rules, such as placing food in 5 rows."
        )
        shot3_q = "What is the difference between Charye and Jesa?"
        shot3_a = (
            "<think>\n"
            "Okay, I need to explain the differences between Charye and Jesa. Let me recall the key distinctions.\n"
            "First, the timing. Charye is for holidays like Seollal or Chuseok, usually in the morning. Jesa is for the ancestor's death anniversary, held late at night or early morning.\n"
            "Second, the formality. Charye is simpler. You might use tea instead of alcohol, and the food is seasonal like tteokguk. The procedures are less complex.\n"
            "Jesa is the formal ritual. It involves offering alcohol (heonjak) and has a strict five-row food arrangement (jinseol).\n"
            "So the main points are: when (holiday vs anniversary), time (day vs night), and complexity (simple vs formal).\n"
            "I should structure the answer to highlight these contrasts clearly, noting that while modern practices blur the lines, the traditional distinction is clear.\n"
            "</think>\n\n"
            "The biggest difference between Charye and Jesa lies in the **timing and time of the ritual, and the formality of the procedures**.\n\n"
            "First, Charye (Tea Ritual) is a simple rite performed on the morning of holidays like Seollal or Chuseok. It is characterized by offering tea instead of alcohol, and offering holiday foods like tteokguk or songpyeon instead of rice and soup. "
            "The procedures are also relatively simple, emphasizing the nature of the family honoring ancestors together.\n\n"
            "On the other hand, Jesa (Gijesa) is a formal ceremony held on the night (around midnight) of the ancestor's death anniversary. It follows complex procedures such as Heonjak (offering alcohol), and the table setting (Jinseol) strictly follows the 5-row principle. "
            "This is the most polite form of showing respect to ancestors.\n\n"
            "In summary, they are clearly distinguished in that Charye is a **'simple rite performed during the day on holidays'**, while Jesa is a **'formal rite performed at night on anniversaries'**. "
            "Although forms are sometimes simplified for convenience in modern times, this distinction is considered important in traditional etiquette."
        )

        # Construct Messages
        messages = [
            {"role": "system", "content": system_prompt},
            
            # Shot 1
            {"role": "user", "content": f"[Background Knowledge]\n{shot1_k}\n\n{shot1_q}"},
            {"role": "assistant", "content": shot1_a},
            
            # Shot 2
            {"role": "user", "content": f"[Background Knowledge]\n{shot2_k}\n\n{shot2_q}"},
            {"role": "assistant", "content": shot2_a},
            
            # Shot 3
            {"role": "user", "content": f"[Background Knowledge]\n{shot3_k}\n\n{shot3_q}"},
            {"role": "assistant", "content": shot3_a},
            
            # Target Input
            {"role": "user", "content": f"[Background Knowledge]\n{knowledge}\n\n{query}"},
        ]
        
        # 3. Tokenize for Teacher (RAG)
        teacher_inputs = self.tokenizer.apply_chat_template(
            messages,
            return_dict=True,
            tokenize=True,
            return_tensors="pt",
            add_generation_prompt=True
        )
        
        # 4. Tokenize for Student (No RAG)
        student_inputs = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": query}],
            return_dict=True,
            tokenize=True,
            return_tensors="pt",
            add_generation_prompt=True
        )
        
        return teacher_inputs, student_inputs

class LUPICollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, padding_side="left"):
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = padding_side # Left padding for generation
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # Separate teacher and student inputs
        t_ids = [item['teacher_input_ids'] for item in batch]
        s_ids = [item['student_input_ids'] for item in batch]
        
        # Pad separately
        t_batch = self.tokenizer.pad(
            {"input_ids": t_ids},
            padding=True,
            return_tensors="pt"
        )
        s_batch = self.tokenizer.pad(
            {"input_ids": s_ids},
            padding=True,
            return_tensors="pt"
        )
        
        return {
            "teacher_input_ids": t_batch['input_ids'],
            "teacher_attention_mask": t_batch['attention_mask'],
            "student_input_ids": s_batch['input_ids'],
            "student_attention_mask": s_batch['attention_mask']
        }
