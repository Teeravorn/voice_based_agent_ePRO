from openthai_systemone import SystemOneClient, Choice, Score, Noul

client = SystemOneClient("iapp/OpenThai-SystemOne")
resp = client.system_one(
    state={"คำตอบผู้ป่วย": "ปวดตลอดเวลาเลยครับ กินยาก็ไม่ค่อยดีขึ้น"},
    questions={
        "map_score": {
            "type": "choice",
            "instructions": "คำถาม: โดยทั่วไป ท่านมีอาการปวดแขน, ไหล่หรือมือ หรือไม่",
            "criteria": {
                "1" : "ไม่มีอาการนี้เลย",
                "2" : "มีอาการเล็กน้อย พอทนได้ ไม่รบกวนการใช้ชีวิต",
                "3" : "มีอาการปานกลาง รู้สึกชัดเจนแต่ยังพอทนได้",
                "4" : "มีอาการรุนแรง รบกวนการใช้ชีวิตประจำวันชัดเจน",
                "5" : "มีอาการรุนแรงมากที่สุด ทนแทบไม่ไหว",
                # "1" : "ไม่มี",
                # "2" : "เล็กน้อย",
                # "3" : "ปานกลาง",
                # "4" : "รุนแรง",
                # "5" : "รุนแรงมากที่สุด",
            },
        }
    },
)
print("Selected:", resp.answers["map_score"].choice, resp.answers["map_score"].probabilities)
# print(resp.answers["frustration"].score, resp.answers["refund_requested"].noul)


# class ChatEngineJevThai:
#     def __init__(self):
#         self.model = "iapp/OpenThai-SystemOne"
#         self.client = SystemOneClient(self.model)

#     def chat(self, state: str, questions: dict):
#         resp = self.client.system_one(state, questions)
#         return resp
