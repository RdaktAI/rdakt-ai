"""Custom patterns — add domain-specific entity types."""

from rdakt_ai import Anonymizer, RdaktConfig, RdaktPipeline, RegexDetector, RegexStage

# Define custom patterns in config
config = RdaktConfig(
    custom_patterns={
        "ACCOUNT_NUMBER": r"\d{4}-\d{4}-\d{4}",
        "INTERNAL_ID": r"EMP-\d{6}",
    },
    entity_strategies={
        "ACCOUNT_NUMBER": "token",
        "INTERNAL_ID": "token",
    },
)

# Build pipeline with custom patterns
detector = RegexDetector(custom_patterns=config.custom_patterns)
pipeline = RdaktPipeline(stages=[RegexStage(detector)])

text = "Employee EMP-123456 has account 1234-5678-9012"
entities = pipeline.detect_sync(text)

print("=== Custom Patterns ===")
print()
print(f"Input: {text}")
print(f"Detected {len(entities)} entities:")
for e in entities:
    print(f"  - {e.type}: '{e.value}'")

anonymizer = Anonymizer()
anonymized, mapping = anonymizer.anonymize(text, entities)
print(f"Anonymized: {anonymized}")
