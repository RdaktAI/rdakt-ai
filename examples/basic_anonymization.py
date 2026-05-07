"""Basic anonymization example — detect and anonymize text directly."""

from rdakt_ai import Anonymizer, RdaktPipeline, RegexDetector, RegexStage

# 1. Set up the detection pipeline
detector = RegexDetector()
pipeline = RdaktPipeline(stages=[RegexStage(detector)])

# 2. Detect entities in text
text = "Contact john@example.com or call (555) 123-4567. SSN: 123-45-6789"
entities = pipeline.detect_sync(text)

print("=== Basic Anonymization ===")
print()
print(f"Input: {text}")
print(f"Detected {len(entities)} entities:")
for e in entities:
    print(f"  - {e.type}: '{e.value}' at [{e.start}:{e.end}]")

# 3. Anonymize
anonymizer = Anonymizer()
anonymized_text, mapping = anonymizer.anonymize(text, entities)

print()
print(f"Anonymized: {anonymized_text}")
print(f"Mapping: {mapping}")
