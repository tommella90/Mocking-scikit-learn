#%%
class DummyClass:
    """"
    A dummy class for testing purposes"""
    def __init__(self):
        self.a = 1
        self.b = 2
        self.c = 3

my_class = DummyClass()

all_attributes = dir(my_class)
all_attributes


# %%
my_class.__dict__

# %%
for i in my_class.__class__.__dict__:
    print(i)

#%%
my_class.__class__.__dict__.__init__

# %%

my_class.__class__.__name__
# %%
my_class.__init__
# %%
my_class.__dict__
# %%
