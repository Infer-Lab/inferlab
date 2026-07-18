---
title: Concepts
description: A map of InferLab's core ownership boundaries and evidence flow.
---

InferLab separates the shareable experiment baseline from machine-private realization facts:

1. A committed **workspace** owns stacks, servers, cases, recipes, and measurement intent.
2. Git-ignored **local bindings** supply concrete machines, devices, ports, model locations, and placement.
3. Resolution produces one effective execution authority used by dry-run, launch, and record production.
4. Non-dry-run workflows write durable **file-first records** containing effective values, lifecycle evidence, metrics, logs, artifacts, and cleanup outcomes.

This page is an orientation aid. [RFC-0001](../architecture/rfc/rfc-0001/) and its topic RFCs remain authoritative for behavior.
