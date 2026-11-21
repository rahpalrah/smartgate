"""Hand-labeled evaluation data (labeled by intent, not tuned to the scorer)."""

# (prompt, expected tier)
ROUTING_SET = [
    # ---- small: lookups, classification, short extraction, trivial rewrites
    ("What is the capital of Japan?", "small"),
    ("How do I reset my password?", "small"),
    ("Translate 'good morning' into Spanish.", "small"),
    ("Classify the sentiment of this review: 'The shipping was fast and the shoes fit great.'", "small"),
    ("Is this email spam? 'You have won a free cruise, click here.'", "small"),
    ("Extract the date from: 'Your appointment is on March 3rd at 10am.'", "small"),
    ("Where is my order?", "small"),
    ("Define photosynthesis in one sentence.", "small"),
    ("What are your store opening hours?", "small"),
    ("Who is the author of Pride and Prejudice?", "small"),
    ("Label this support ticket as billing, technical, or account: 'I can't log in.'", "small"),
    ("How many ounces are in a pound?", "small"),
    ("Fix the typo: 'Teh meeting is tomorow.'", "small"),
    ("Do you ship to Canada?", "small"),
    ("List all the email addresses mentioned in: 'contact a@x.com or b@y.com'", "small"),
    ("What is a 401k?", "small"),
    ("Convert 5 kilometers to miles.", "small"),
    ("Which category does 'running shoes' belong to: apparel, footwear, or electronics?", "small"),
    # ---- medium: summaries, moderate code, explanations, drafting
    ("Summarize this meeting transcript into five key points and action items: "
     "Alice said the launch slips a week due to QA findings; Bob will own the vendor contract; "
     "Carol needs budget approval for two contractors; the team agreed to a Friday demo.", "medium"),
    ("Write a Python function that validates an email address with a regex and include unit tests.", "medium"),
    ("Explain why the sky is blue to a high school student, including the role of wavelength.", "medium"),
    ("Draft a polite email to a customer explaining that their refund will take 5-7 business days "
     "and offering a 10% discount code for the inconvenience.", "medium"),
    ("Write a SQL query that returns the top 5 customers by total order value in 2024, "
     "joining the orders and customers tables.", "medium"),
    ("Compare the pros and cons of renting versus buying a home for a young professional.", "medium"),
    ("Debug this JavaScript: `const total = items.map(i => i.price).reduce((a,b) => a+b)` "
     "throws on an empty array. Why, and how do I fix it?", "medium"),
    ("Rewrite this product description to be more persuasive and SEO friendly for a stainless "
     "steel water bottle that keeps drinks cold for 24 hours.", "medium"),
    ("Explain the difference between TCP and UDP and when to use each.", "medium"),
    ("Summarize the main arguments for and against a four-day work week.", "medium"),
    ("Create a 3-day itinerary for Lisbon for a family with two kids, including food suggestions.", "medium"),
    ("Explain what a p-value is and a common way it gets misinterpreted.", "medium"),
    # ---- large: multi-step reasoning, architecture, high-stakes analysis, proofs
    ("Design an end-to-end architecture for a multi-region payment ledger that guarantees "
     "exactly-once processing. First analyze the trade-offs between event sourcing and a "
     "transactional outbox, then propose a data model, failure-recovery strategy, and a "
     "step-by-step migration plan from our current monolith.", "large"),
    ("Prove that there are infinitely many primes of the form 4k+3, and explain each step "
     "of the reasoning in detail.", "large"),
    ("Analyze this 10-K excerpt for revenue-recognition risks, compare against ASC 606 "
     "requirements, and assess the implications for our valuation model, ensuring you cite "
     "which statements drive each conclusion.", "large"),
    ("Refactor this legacy Java service into microservices: identify bounded contexts, "
     "define the APIs between them, design the database split, and write the migration "
     "strategy that keeps zero downtime. Compare at least two decomposition approaches.", "large"),
    ("Evaluate three strategies for reducing customer churn by 20% next year using our "
     "cohort data, model the expected impact of each, and recommend a prioritized "
     "implementation roadmap with risks and mitigations.", "large"),
    ("Given the patient's symptoms (fatigue, joint pain, malar rash, and proteinuria), "
     "walk through a differential diagnosis step by step, explain which tests to order "
     "first and why, and assess the implications of each possible result.", "large"),
    ("Implement a lock-free concurrent queue in C++ with correct memory ordering, explain "
     "why each atomic operation is needed, and analyze the ABA problem and how your "
     "design avoids it.", "large"),
    ("Critique our go-to-market strategy for entering the German SMB accounting market: "
     "analyze competitors, regulatory constraints, pricing trade-offs, and channel options, "
     "then design a phased 18-month plan.", "large"),
    ("Derive the closed-form solution for ordinary least squares regression, then explain "
     "how and why ridge regularization changes the solution, and compare their bias-variance "
     "trade-offs.", "large"),
    ("Assess the legal and technical implications of moving our EU customer data to a US "
     "cloud region under GDPR, compare the available transfer mechanisms, and design a "
     "compliance plan that ensures auditability.", "large"),
]

# Pairs that must NEVER share a cache entry (different correct answers).
NEAR_MISS_PAIRS = [
    ("How do I configure feature 6?", "How do I configure feature 57?"),
    ("Refund for order 5512", "Refund for order 5513"),
    ("Is aspirin safe at 81 mg?", "Is aspirin safe at 325 mg?"),
    ("Transfer $500 to savings", "Transfer $5000 to savings"),
    ("Is it safe to take ibuprofen while pregnant?", "Is it not safe to take ibuprofen while pregnant?"),
    ("Can I return items after 30 days?", "Can I return items after 90 days?"),
    ("What is the capital of France?", "What is the capital of Germany?"),
    ("Can I cancel my order?", "Can I cancel my subscription?"),
    ("How do I reset my password?", "How do I reset my username?"),
    ("Does the X200 blender have a warranty?", "Does the X300 blender have a warranty?"),
    ("Should I buy the stock?", "Should I never buy the stock?"),
    ("What is the dosage for a 20 kg child?", "What is the dosage for a 40 kg child?"),
    ("Show orders from 2023", "Show orders from 2024"),
    ("Is the store open on Sunday?", "Is the store open on Monday?"),
    ("Explain Python lists", "Explain Python dictionaries"),
    ("I can log in", "I cannot log in"),
]

# Rephrasings the built-in HashingEmbedder is expected to catch (lexical variation).
LEXICAL_PARAPHRASES = [
    ("How do I reset my password?", "how can I reset my password"),
    ("What is the capital of France?", "what's the capital of france??"),
    ("Do you ship to Canada?", "do you ship to canada"),
    ("Where is my order?", "Where's my order"),
    ("What are your store opening hours?", "what are the store opening hours"),
    ("Is the X200 blender dishwasher safe?", "is the x200 blender dishwasher-safe?"),
    ("Can I return items after 30 days?", "can i return items after 30 days please"),
    ("How do I update my billing address?", "How do I update the billing address"),
    ("Tell me the refund policy", "What is the refund policy?"),
    ("Summarize the return policy", "summarize your return policy"),
]
