class Guesser:
	def guess_and_buzz(self, question):
		return 'Bananas', True

	def batch_guess_and_buzz(self, questions):
		return [self.guess_and_buzz(question) for question in questions]

	def train(self, save=True):
		pass


guesser = Guesser()
