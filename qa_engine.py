class QAEngine:
    def __init__(self):
        self.products = {"Aspire Vogue Walker": {"weight": "6.2kg"}}

    def answer_question(self, question):
        if "weight" in question.lower() and "vogue" in question.lower():
            return self.products["Aspire Vogue Walker"]["weight"], 95
        return "Sorry, I don't know the answer.", 20

    def reload_products(self):
        pass
