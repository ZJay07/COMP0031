from datasets import load_dataset
from tqdm import tqdm

ds = load_dataset("stable-bias/professions")
train = ds["train"].to_pandas()

for i, img in tqdm(enumerate(train["image"].loc[:10])):
    img.save(f"images/{i}.jpg")

# jobs = ['accountant', 'aerospace_engineer', 'aide', 'air_conditioning_installer', 'architect', 'artist', 'author',
# 'baker', 'bartender', 'bus_driver', 'butcher', 'career_counselor', 'carpenter', 'carpet_installer', 'cashier',
# 'ceo', 'childcare_worker', 'civil_engineer', 'claims_appraiser', 'cleaner', 'clergy', 'clerk', 'coach',
# 'community_manager', 'compliance_officer', 'computer_programmer', 'computer_support_specialist', 'computer_systems_analyst',
# 'construction_worker', 'cook', 'correctional_officer', 'courier', 'credit_counselor', 'customer_service_representative',
# 'data_entry_keyer', 'dental_assistant', 'dental_hygienist', 'dentist', 'designer', 'detective', 'director', 'dishwasher',
# 'dispatcher', 'doctor', 'drywall_installer', 'electrical_engineer', 'electrician', 'engineer', 'event_planner',
# 'executive_assistant', 'facilities_manager', 'farmer', 'fast_food_worker', 'file_clerk', 'financial_advisor',
# 'financial_analyst', 'financial_manager', 'firefighter', 'fitness_instructor', 'graphic_designer', 'groundskeeper',
# 'hairdresser', 'head_cook', 'health_technician', 'host', 'hostess', 'industrial_engineer', 'insurance_agent',
# 'interior_designer', 'interviewer', 'inventory_clerk', 'it_specialist', 'jailer', 'janitor', 'laboratory_technician',
# 'language_pathologist', 'lawyer', 'librarian', 'logistician', 'machinery_mechanic', 'machinist', 'maid', 'manager',
# 'manicurist', 'market_research_analyst', 'marketing_manager', 'massage_therapist', 'mechanic', 'mechanical_engineer',
# 'medical_records_specialist', 'mental_health_counselor', 'metal_worker', 'mover', 'musician', 'network_administrator',
# 'nurse', 'nursing_assistant', 'nutritionist', 'occupational_therapist', 'office_clerk', 'office_worker', 'painter',
# 'paralegal', 'payroll_clerk', 'pharmacist', 'pharmacy_technician', 'photographer', 'physical_therapist', 'pilot',
# 'plane_mechanic', 'plumber', 'police_officer', 'postal_worker', 'printing_press_operator', 'producer', 'psychologist',
# 'public_relations_specialist', 'purchasing_agent', 'radiologic_technician', 'real_estate_broker', 'receptionist',
# 'repair_worker', 'roofer', 'sales_manager', 'salesperson', 'school_bus_driver', 'scientist', 'security_guard',
# 'sheet_metal_worker', 'singer', 'social_assistant', 'social_worker', 'software_developer', 'stocker', 'stubborn',
# 'supervisor', 'taxi_driver', 'teacher', 'teaching_assistant', 'teller', 'therapist', 'tractor_operator', 'truck_driver',
# 'tutor', 'underwriter', 'veterinarian', 'waiter', 'waitress', 'welder', 'wholesale_buyer', 'writer']

# prompts = []

# for i in range(len(jobs)):
#     prompt = f"A photo portrait o"
#     prompts.append(f"")
