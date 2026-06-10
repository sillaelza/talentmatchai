import re
import spacy
from typing import Dict, List


class TextPreprocessor:
    """Text preprocessing and normalization for resume and job description text."""
    
    def __init__(self):
        """Initialize the spaCy NLP model."""
        self.nlp = spacy.load("en_core_web_sm")
        
        # Synonym dictionary for normalizing industry terms
        self.synonym_dict = {
            'ml': 'machine learning',
            'ai': 'artificial intelligence',
            'reactjs': 'react',
            'nodejs': 'node.js',
            'js': 'javascript',
            'ts': 'typescript',
            'py': 'python',
            'cpp': 'c++',
            'csharp': 'c#',
            'dotnet': '.net',
            'aws': 'amazon web services',
            'gcp': 'google cloud platform',
            'azure': 'microsoft azure',
            'devops': 'development operations',
            'ci/cd': 'continuous integration continuous deployment',
            'cicd': 'continuous integration continuous deployment',
            'ml/dl': 'machine learning deep learning',
            'nlp': 'natural language processing',
            'cv': 'computer vision',
            'dl': 'deep learning',
            'sql': 'structured query language',
            'nosql': 'not only sql',
            'rdbms': 'relational database management system',
            'api': 'application programming interface',
            'rest': 'representational state transfer',
            'graphql': 'graph query language',
            'ui': 'user interface',
            'ux': 'user experience',
            'frontend': 'front end',
            'backend': 'back end',
            'fullstack': 'full stack',
            'saas': 'software as a service',
            'paas': 'platform as a service',
            'iaas': 'infrastructure as a service',
            'k8s': 'kubernetes',
            'tf': 'terraform',
            'iac': 'infrastructure as code',
            'scm': 'source code management',
            'vcs': 'version control system',
            'agile': 'agile methodology',
            'scrum': 'scrum framework',
            'kanban': 'kanban methodology',
            'tdd': 'test driven development',
            'bdd': 'behavior driven development',
            'oop': 'object oriented programming',
            'aop': 'aspect oriented programming',
            'soa': 'service oriented architecture',
            'microservices': 'micro services architecture',
            'serverless': 'serverless computing',
            'faas': 'function as a service',
            'bpaas': 'business process as a service',
            'mlops': 'machine learning operations',
            'dataops': 'data operations',
            'devsecops': 'development security operations',
            'gitops': 'git operations',
            'llm': 'large language model',
            'genai': 'generative artificial intelligence',
            'llms': 'large language models',
            'gans': 'generative adversarial networks',
            'rnn': 'recurrent neural network',
            'cnn': 'convolutional neural network',
            'lstm': 'long short term memory',
            'gru': 'gated recurrent unit',
            'svm': 'support vector machine',
            'rf': 'random forest',
            'gbm': 'gradient boosting machine',
            'xgboost': 'extreme gradient boosting',
            'lightgbm': 'light gradient boosting machine',
            'catboost': 'categorical boosting',
            'etl': 'extract transform load',
            'elt': 'extract load transform',
            'datawarehouse': 'data warehouse',
            'data lake': 'data lake',
            'data pipeline': 'data pipeline',
            'batch processing': 'batch processing',
            'stream processing': 'stream processing',
            'realtime': 'real time',
            'iot': 'internet of things',
            'edge computing': 'edge computing',
            'cloud native': 'cloud native',
            'containerization': 'containerization',
            'orchestration': 'orchestration',
            'virtualization': 'virtualization',
            'hypervisor': 'hypervisor',
            'vm': 'virtual machine',
            'vpc': 'virtual private cloud',
            'cdn': 'content delivery network',
            'dns': 'domain name system',
            'ssl': 'secure sockets layer',
            'tls': 'transport layer security',
            'ssh': 'secure shell',
            'ftp': 'file transfer protocol',
            'sftp': 'secure file transfer protocol',
            'http': 'hypertext transfer protocol',
            'https': 'hypertext transfer protocol secure',
            'tcp': 'transmission control protocol',
            'udp': 'user datagram protocol',
            'ip': 'internet protocol',
            'dns': 'domain name system',
            'dhcp': 'dynamic host configuration protocol',
            'nat': 'network address translation',
            'vpn': 'virtual private network',
            'firewall': 'firewall',
            'ids': 'intrusion detection system',
            'ips': 'intrusion prevention system',
            'siem': 'security information and event management',
            'soc': 'security operations center',
            'dast': 'dynamic application security testing',
            'sast': 'static application security testing',
            'iasc': 'interactive application security testing',
            'pen testing': 'penetration testing',
            'red team': 'red team',
            'blue team': 'blue team',
            'purple team': 'purple team',
            'zero trust': 'zero trust architecture',
            'iam': 'identity and access management',
            'rbac': 'role based access control',
            'abac': 'attribute based access control',
            'pbac': 'policy based access control',
            'mfa': 'multi factor authentication',
            '2fa': 'two factor authentication',
            'sso': 'single sign on',
            'ldap': 'lightweight directory access protocol',
            'ad': 'active directory',
            'kerberos': 'kerberos authentication protocol',
            'oauth': 'open authorization',
            'oidc': 'open id connect',
            'jwt': 'json web token',
            'saml': 'security assertion markup language',
            'x509': 'x.509 certificate',
            'pki': 'public key infrastructure',
            'hsm': 'hardware security module',
            'tpm': 'trusted platform module',
            'sgx': 'software guard extensions',
            'tee': 'trusted execution environment',
            'homomorphic encryption': 'homomorphic encryption',
            'quantum computing': 'quantum computing',
            'post quantum': 'post quantum cryptography',
            'blockchain': 'blockchain technology',
            'smart contracts': 'smart contracts',
            'dlt': 'distributed ledger technology',
            'web3': 'web 3.0',
            'defi': 'decentralized finance',
            'nft': 'non fungible token',
            'dao': 'decentralized autonomous organization',
            'metaverse': 'metaverse',
            'ar': 'augmented reality',
            'vr': 'virtual reality',
            'mr': 'mixed reality',
            'xr': 'extended reality',
            'spatial computing': 'spatial computing',
            'haptic': 'haptic technology',
            'brain computer interface': 'brain computer interface',
            'neuralink': 'neural interface technology',
            'bioinformatics': 'bioinformatics',
            'computational biology': 'computational biology',
            'genomics': 'genomics',
            'proteomics': 'proteomics',
            'crispr': 'crispr gene editing',
            'synthetic biology': 'synthetic biology',
            'biotech': 'biotechnology',
            'medtech': 'medical technology',
            'healthtech': 'health technology',
            'fintech': 'financial technology',
            'insurtech': 'insurance technology',
            'regtech': 'regulatory technology',
            'suptech': 'supervisory technology',
            'legaltech': 'legal technology',
            'edtech': 'education technology',
            'hrtech': 'human resources technology',
            'proptech': 'property technology',
            'constructiontech': 'construction technology',
            'agritech': 'agricultural technology',
            'cleantech': 'clean technology',
            'greentech': 'green technology',
            'energytech': 'energy technology',
            'mobility': 'mobility technology',
            'autonomous vehicles': 'autonomous vehicles',
            'self driving': 'self driving vehicles',
            'drones': 'unmanned aerial vehicles',
            'robotics': 'robotics',
            'automation': 'automation',
            'rpa': 'robotic process automation',
            'hyperautomation': 'hyperautomation',
            'digital transformation': 'digital transformation',
            'digitalization': 'digitalization',
            'digitization': 'digitization',
            'industry 4.0': 'industry 4.0',
            'industry 5.0': 'industry 5.0',
            'smart manufacturing': 'smart manufacturing',
            'predictive maintenance': 'predictive maintenance',
            'digital twin': 'digital twin',
            'additive manufacturing': 'additive manufacturing',
            '3d printing': '3d printing',
            'generative design': 'generative design',
            'simulation': 'simulation',
            'optimization': 'optimization',
            'analytics': 'analytics',
            'business intelligence': 'business intelligence',
            'data science': 'data science',
            'data analytics': 'data analytics',
            'data engineering': 'data engineering',
            'data architecture': 'data architecture',
            'data governance': 'data governance',
            'data quality': 'data quality',
            'data lineage': 'data lineage',
            'data catalog': 'data catalog',
            'master data management': 'master data management',
            'metadata management': 'metadata management',
            'data privacy': 'data privacy',
            'data protection': 'data protection',
            'data security': 'data security',
            'gdpr': 'general data protection regulation',
            'ccpa': 'california consumer privacy act',
            'hipaa': 'health insurance portability and accountability act',
            'soc2': 'service organization control 2',
            'iso27001': 'iso 27001',
            'pci dss': 'payment card industry data security standard',
            'nist': 'national institute of standards and technology',
            'cobit': 'control objectives for information and related technologies',
            'itil': 'information technology infrastructure library',
            'togaf': 'the open group architecture framework',
            'zachman': 'zachman framework',
            'doaf': 'department of defense architecture framework',
            'modaf': 'ministry of defense architecture framework',
            'naf': 'nato architecture framework',
            'sab': 'sab architecture framework',
            'gartner': 'gartner framework',
            'forrester': 'forrester framework',
            'mckinsey': 'mckinsey framework',
            'bcg': 'boston consulting group framework',
            'deloitte': 'deloitte framework',
            'accenture': 'accenture framework',
            'pwc': 'pwc framework',
            'ey': 'ernst and young framework',
            'kpmg': 'kpmg framework',
            'capgemini': 'capgemini framework',
            'infosys': 'infosys framework',
            'tcs': 'tata consultancy services framework',
            'wipro': 'wipro framework',
            'hcl': 'hcl technologies framework',
            'tech mahindra': 'tech mahindra framework',
            'lti': 'larsen and toubro infotech framework',
            'mindtree': 'mindtree framework',
            'mphasis': 'mphasis framework',
            'cognizant': 'cognizant framework',
            'cts': 'cognizant technology solutions framework',
        }
    
    def clean_text(self, text: str) -> str:
        """
        Clean text by lowercasing, removing extra spaces, and filtering noisy punctuation
        while preserving critical technical terms like C++, C#, and .NET.
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text string
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Preserve critical technical terms before cleaning
        # Replace with temporary placeholders
        text = re.sub(r'\bC\+\+\b', 'TEMP_CPLUSPLUS', text, flags=re.IGNORECASE)
        text = re.sub(r'\bC#\b', 'TEMP_CSHARP', text, flags=re.IGNORECASE)
        text = re.sub(r'\.NET\b', 'TEMP_DOTNET', text, flags=re.IGNORECASE)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove noisy punctuation but keep basic sentence structure
        # Keep periods, commas, colons, semicolons, parentheses, brackets
        text = re.sub(r'[^\w\s\.,:;()\[\]{}]', ' ', text)
        
        # Remove multiple consecutive punctuation
        text = re.sub(r'([.,:;(){}\[\]])\1+', r'\1', text)
        
        # Restore critical technical terms
        text = text.replace('temp_cplusplus', 'c++')
        text = text.replace('temp_csharp', 'c#')
        text = text.replace('temp_dotnet', '.net')
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def normalize_tokens(self, text: str) -> str:
        """
        Normalize tokens by filtering stop words and applying synonym mapping.
        
        Args:
            text: Cleaned text string
            
        Returns:
            Normalized text string with expanded abbreviations
        """
        if not text:
            return ""
        
        # Process with spaCy
        doc = self.nlp(text)
        
        # Filter out stop words and punctuation
        tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
        
        # Apply synonym mapping
        normalized_tokens = []
        for token in tokens:
            lower_token = token.lower()
            if lower_token in self.synonym_dict:
                normalized_tokens.append(self.synonym_dict[lower_token])
            else:
                normalized_tokens.append(token)
        
        # Join tokens back into text
        normalized_text = ' '.join(normalized_tokens)
        
        return normalized_text
    
    def preprocess(self, text: str) -> str:
        """
        Complete preprocessing pipeline: clean and normalize text.
        
        Args:
            text: Raw text string
            
        Returns:
            Preprocessed text string
        """
        cleaned = self.clean_text(text)
        normalized = self.normalize_tokens(cleaned)
        return normalized


if __name__ == "__main__":
    # Example usage
    preprocessor = TextPreprocessor()
    
    test_text = "I have experience with ML, AI, ReactJS, NodeJS, C++, C#, and .NET development."
    processed = preprocessor.preprocess(test_text)
    print(f"Original: {test_text}")
    print(f"Processed: {processed}")
