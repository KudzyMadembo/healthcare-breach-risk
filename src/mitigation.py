"""Mitigation recommendation utilities for predicted attack outcomes."""

from __future__ import annotations


PRIORITY_BY_RISK_LEVEL = {
    "Critical": "immediate response",
    "High": "urgent investigation",
    "Medium": "analyst review",
    "Low": "monitor",
}


def _normalize_risk_level(risk_level: str) -> str:
    """Normalize risk-level text to expected canonical values."""
    normalized = risk_level.strip().lower()
    mapping = {
        "critical": "Critical",
        "high": "High",
        "medium": "Medium",
        "low": "Low",
    }
    if normalized not in mapping:
        raise ValueError("risk_level must be one of: Critical, High, Medium, Low")
    return mapping[normalized]


def _action_for_class(predicted_class: str) -> str:
    """Return recommended action based on predicted attack class family."""
    cls = predicted_class.strip().lower()

    if "benign" in cls:
        return "No immediate containment needed; continue routine monitoring."

    if "arp" in cls and "spoof" in cls:
        return (
            "Inspect LAN segment for ARP poisoning indicators and isolate the suspicious host."
        )

    if "mqtt" in cls and ("flood" in cls or "malformed" in cls):
        return (
            "Rate limit MQTT broker traffic and inspect IoMT broker logs for abuse patterns."
        )

    if "recon" in cls or "scan" in cls:
        return (
            "Block the scanning source, review exposed services, and validate segmentation rules."
        )

    if any(token in cls for token in ("dos", "ddos")) and any(
        token in cls for token in ("tcp", "udp", "icmp", "syn")
    ):
        return (
            "Apply rate limits and traffic filters, isolate the target path, and alert SOC for incident handling."
        )

    return (
        "Investigate anomalous traffic, validate detection context, and escalate if impact indicators increase."
    )


def recommend_action(predicted_class: str, risk_level: str) -> dict[str, str]:
    """Map predicted class and risk level to mitigation guidance.

    Args:
        predicted_class: Predicted attack class label.
        risk_level: One of Critical, High, Medium, Low.

    Returns:
        Dictionary with:
        - predicted_class
        - risk_level
        - recommended_action
        - priority
    """
    normalized_risk = _normalize_risk_level(risk_level)
    priority = PRIORITY_BY_RISK_LEVEL[normalized_risk]
    action = _action_for_class(predicted_class)

    return {
        "predicted_class": predicted_class,
        "risk_level": normalized_risk,
        "recommended_action": action,
        "priority": priority,
    }


if __name__ == "__main__":
    test_cases = [
        ("ARP_Spoofing", "Critical"),
        ("MQTT-DDoS-Publish_Flood", "High"),
        ("Recon-Port_Scan", "Medium"),
        ("TCP_IP-DDoS-UDP", "Critical"),
        ("Benign", "Low"),
    ]

    print("Mitigation recommendation examples:")
    for predicted_class, risk_level in test_cases:
        recommendation = recommend_action(predicted_class, risk_level)
        print(recommendation)
