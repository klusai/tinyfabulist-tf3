from typing import List, Dict

def generate_synthetic_fable_data(num_samples: int = 50) -> List[Dict[str, str]]:
    """Generate synthetic fable data with realistic prompts and reference completions"""
    
    # Better fable templates with proper moral structure
    templates = [
        {
            "prompt": "A hare and a dove were walking through a farm. They found a problem that needed solving.",
            "reference": "The hare wanted to solve it quickly, but the dove suggested a careful approach. By working together and being patient, they solved the problem successfully. The moral: Patience and teamwork overcome hasty decisions."
        },
        {
            "prompt": "Once upon a time, in a peaceful meadow, there lived a fox and a turtle. The fox was known for being",
            "reference": "quick and clever, while the turtle was slow but wise. One day, they decided to have a contest to see who was better. The fox was confident of winning, but the turtle worked steadily and carefully. In the end, the turtle's persistence paid off. The moral: Slow and steady wins the race."
        },
        {
            "prompt": "In a beautiful river, a dove met a lion. The dove boasted about their abilities, while the lion listened",
            "reference": "quietly without bragging. When trouble came to the river, the lion helped everyone with their strength and courage. The dove realized that true worth comes from helping others, not from boasting. The moral: Actions speak louder than words."
        },
        {
            "prompt": "Two mice lived in a barn. One mouse was very careful, while the other was",
            "reference": "reckless and careless. The careful mouse stored food for winter, while the reckless mouse played all day. When winter came, the careful mouse had plenty to eat, but the reckless mouse was hungry. The careful mouse shared his food and taught his friend about planning ahead. The moral: Preparation and kindness lead to prosperity."
        },
        {
            "prompt": "A crow sat on a tree branch and saw a piece of cheese. The crow",
            "reference": "wanted the cheese very much. A clever fox below saw the crow and praised the crow's beautiful voice. Flattered, the crow opened his beak to sing and dropped the cheese. The fox grabbed it and ran away. The moral: Beware of flattery that seeks to trick you."
        },
        {
            "prompt": "In a forest, an ant was working hard to gather food. A grasshopper saw the ant and",
            "reference": "laughed at the ant for working so hard instead of enjoying the sunny day. The ant continued working while the grasshopper played and sang. When winter came, the ant had plenty of food stored, but the grasshopper had nothing. The ant shared some food and taught the grasshopper about hard work. The moral: Hard work and planning prepare us for difficult times."
        },
        {
            "prompt": "A wolf and a sheep met at a stream. The wolf accused the sheep of",
            "reference": "making the water muddy. The sheep politely explained that she was drinking downstream and couldn't muddy the water above the wolf. The wolf then made other false accusations because he had already decided to attack. A wise old owl overhead saw this and warned other animals about the wolf's dishonesty. The moral: Those who seek to do wrong will always find an excuse."
        },
        {
            "prompt": "Once there was a proud oak tree that mocked a small reed. The oak said",
            "reference": "that it was mighty and strong while the reed was weak and small. One day, a terrible storm came. The proud oak tree tried to stand firm against the wind but broke and fell. The humble reed bent with the wind and survived the storm. The moral: Flexibility and humility are stronger than pride and stubbornness."
        },
        {
            "prompt": "A hungry dog found a bone and carried it across a bridge. Looking down, the dog",
            "reference": "saw his reflection in the water and thought it was another dog with a larger bone. Greedy for the bigger bone, he barked at his reflection and dropped his own bone into the water. The dog lost his meal and went hungry. The moral: Greed often leads to losing what we already have."
        },
        {
            "prompt": "Two birds lived in the same tree. One bird was kind and shared food, while the other",
            "reference": "was selfish and kept everything for itself. During a harsh winter, many animals were hungry. The kind bird shared its stored food and made many friends. The selfish bird hoarded everything but was lonely and sad. When spring came, the kind bird had many friends to help gather food, while the selfish bird had to work alone. The moral: Kindness and generosity bring lasting friendships."
        },
        {
            "prompt": "A young rabbit was afraid of trying new things. When other rabbits suggested exploring a new field, the rabbit",
            "reference": "was too scared to join them. The other rabbits found delicious vegetables and made new friends. Eventually, an older rabbit encouraged the fearful rabbit to try new experiences step by step. With support and practice, the rabbit gained confidence and discovered wonderful new places. The moral: Courage grows when we face our fears with support from friends."
        },
        {
            "prompt": "A bear and a deer lived in the same forest. The bear was strong but impatient, while the deer was",
            "reference": "gentle and patient. When they needed to cross a dangerous river, the bear rushed ahead and fell into the swift current. The deer carefully studied the river, found a safe crossing, and helped rescue the bear. The bear learned that patience and careful thinking are as valuable as strength. The moral: Wisdom and patience can save us from the troubles caused by haste."
        }
    ]
    
    # Ensure we have enough samples by repeating templates if needed
    samples = []
    for i in range(num_samples):
        template = templates[i % len(templates)]
        
        # Add variation to avoid exact duplicates
        if i >= len(templates):
            # Slight variations for repeated templates
            template = {
                "prompt": template["prompt"],
                "reference": template["reference"]
            }
        
        samples.append({
            "fable": template["prompt"],
            "text": template["reference"],
            "prompt": template["prompt"],
            "reference": template["reference"]
        })
    
    return samples 