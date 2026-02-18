# Meishi CRM

Japanese Business Card OCR + Structured Contact Extraction

## Overview

Meishi CRM is a lightweight contact management system that extracts structured data from Japanese business cards using OCR and rule-based parsing with optional LLM enhancement.

The system:

• Extracts Japanese and English text using Tesseract OCR
• Parses Name, Company, Title, Email, Phone
• Falls back to an LLM when rule-based extraction is incomplete
• Deduplicates contacts by email or phone
• Stores structured records in Excel
• Runs as a Dockerized FastAPI web application

Designed as an end-to-end ML prototype covering OCR, NLP parsing, backend API, and containerized deployment.

---

## Features

• Japanese + English OCR (Tesseract jpn + eng)
• Hybrid parsing pipeline

* Rule-based extraction for deterministic fields
* LLM fallback using TinyLlama for missing structured fields
  • Idempotent upsert logic (no duplicate contacts)
  • REST API built with FastAPI
  • Simple web UI for upload
  • Dockerized for reproducible deployment

---

## Key Design Decisions

Hybrid Parsing
Rule-based logic handles structured patterns reliably.
LLM fallback improves recall when OCR output is noisy or layout is complex.

Deduplication Strategy
Contacts are matched on Email first, then Phone.
Prevents duplicate records in master.xlsx.

Lightweight Storage
Excel chosen for portability and non-technical user compatibility.

Containerization
Docker ensures reproducible environment with OCR + ML dependencies.

---

## Limitations

• OCR accuracy depends on image quality
• LLM model is lightweight and may misformat JSON
• Excel storage not optimized for high concurrency
• CPU inference may increase latency

---

## Project Purpose

This project demonstrates:

• End-to-end ML system design
• OCR + NLP integration
• Hybrid rule-based + LLM extraction strategy
• Backend API development
• Dockerized deployment
• Data deduplication logic

It reflects practical system building beyond model experimentation, focusing on deployable architecture and data handling reliability.

---

## Author

Aditya Agrawal
Machine Learning Engineer
IIT Delhi – Mathematics & Computing