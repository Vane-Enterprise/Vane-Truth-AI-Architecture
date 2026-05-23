Vane-Truth-AI-Architecture

Enforcing the Law of the Diamond to Eliminate AI Drift

The Vane-Truth-AI-Architecture is the authoritative repository for the Vane-Guard Sovereign Framework (v1.0). This enterprise-grade Truth-AI system is designed to eliminate hallucinations and "AI drift" in infrastructure diagnostics by anchoring all reasoning to verified technical documentation.

💎 The Law of the Diamond

The framework operates through four mandatory logic gates to ensure every diagnostic report is an authenticated "Output of Privilege":
1. Identity Verification: All requests are hard-locked to the VANE_ROOT_STABLE_001 identity anchor. 
2. Test Selection: Specialized diagnostic modules for Cloud (AWS/Entra), Networking (STP/HSRP), Application (JVM), and Hardware (HP/Dell). 
3. Report Generation: A proprietary RAG pipeline constructs a 100% transparent Evidence Trail, citing authorized sources for every technical fact. 
4. Interpretation: Translating raw telemetry into actionable insights with a verified Confidence Assessment (HIGH, MEDIUM, or LOW).

🛠️ Technical Specifications

LLM Lock: Mandatory 0.0 Temperature to eliminate creativity and ensure "Ground Truth" precision. 
Security Boundary: Private root identifiers are managed exclusively via GitHub Secrets and never committed in plaintext. 
Identity Anchor: Verified domain synchronization through taruglobalaccess.com

📂 Core Architecture

app.py: The web orchestrator and diagnostic dashboard. 
config.py: Central configuration for branding and identity anchoring. 
llm_provider.py: The truth orchestrator enforcing specialized system prompts. 
rag_pipeline.py: The engine responsible for data retrieval and evidence trail construction. 
chroma_db/: Authorized vector store for technical telemetry.

🚀 Getting Started

1. Requirements

License: Professional license required via the Vane-Guard Store (https://dantevane.gumroad.com/l/Vane-Guard). 
Secrets: Add VANE_ROOT_ID and GITHUB_PRIVATE_KEY to my repository's GitHub Secrets

2. Implementation

Initialize my orchestrator by syncing my environment variables:
import os 
VANE_ROOT_ID = os.getenv("VANE_ROOT_ID") # Hard-locks the reasoning chain 

📞 Support & Community

Authorized Store: Vane-Guard Sovereign Framework(v1.0). (https://dantevane.gumroad.com/l/Vane-Guard) 

Customer Support: Official Vane Enterprise Support (https://www.facebook.com/mdabulhossain1008) 

Company: Vane Enterprise LLC
