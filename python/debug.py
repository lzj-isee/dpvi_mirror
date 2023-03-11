class TEST:
    def __init__(self) -> None:
        self.name = 1

    def func(self):
        pass

print(TEST().__getattribute__('func'))
print(TEST().__getattribute__('name'))