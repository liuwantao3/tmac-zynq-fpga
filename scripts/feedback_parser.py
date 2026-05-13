#!/usr/bin/env python3
"""
Three-Layer FPGA Feedback Parser
Layer 1: Console stdout/stderr (from logs)
Layer 2: Report files (HLS synthesis, Vivado utilization)
Layer 3: Tcl query interface (Vivado interactive)

Output: Structured JSON with metrics and recommendations
"""

import json
import os
import re
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

WORKSPACE = Path("/Users/arctic/fpga")
ZYNQ_7010 = {
    "DSP": 80,
    "BRAM_KB": 240,
    "LUT": 17600,
    "FF": 35200
}

class FeedbackParser:
    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.results = {
            "timestamp": "",
            "hls": {},
            "vivado_util": {},
            "vivado_timing": {},
            "recommendations": []
        }

    def parse_hls_reports(self) -> Dict:
        """Layer 2: Parse HLS synthesis reports"""
        hls_data = {
            "latency_cycles": None,
            "latency_us": None,
            "DSP_used": 0,
            "DSP_percent": 0,
            "LUT_used": 0,
            "LUT_percent": 0,
            "BRAM_used": 0,
            "BRAM_percent": 0,
            "FF_used": 0,
            "FF_percent": 0,
            "status": "unknown"
        }

        hls_dir = WORKSPACE / "hls"
        solution_dir = hls_dir / "solution"

        if not solution_dir.exists():
            return hls_data

        for report_file in ["solution.xml", "solution_report.xml"]:
            report_path = solution_dir / report_file
            if report_path.exists():
                content = report_path.read_text()
                hls_data.update(self._extract_hls_metrics(content))
                break

        for cs_report in solution_dir.glob("csynth.xml"):
            content = cs_report.read_text()
            hls_data.update(self._extract_hls_metrics(content))
            break

        return hls_data

    def _extract_hls_metrics(self, content: str) -> Dict:
        metrics = {}
        patterns = {
            "latency_cycles": r'<Latency>\s*<ClockPeriod>\s*([\d.]+)',
            "DSP_used": r'<DSP>([\d]+)',
            "LUT_used": r'<LUT>([\d]+)',
            "BRAM_used": r'<BRAM>([\d]+)',
            "FF_used": r'<FF>([\d]+)',
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                metrics[key] = int(match.group(1))

        interval_match = re.search(r'<Interval>\s*<ClockPeriod>\s*([\d.]+)', content, re.IGNORECASE)
        if interval_match:
            metrics["interval_cycles"] = int(interval_match.group(1))

        return metrics

    def parse_vivado_reports(self) -> Tuple[Dict, Dict]:
        """Layer 2: Parse Vivado implementation reports"""
        vivado_dir = WORKSPACE / "vivado"
        util_data = {"utilized": {}, "available": {}, "percent": {}}
        timing_data = {"wns": None, "tns": None, "ws": None}

        util_file = vivado_dir / "design_2_utilization_routed.rpt"
        if util_file.exists():
            content = util_file.read_text()
            util_data = self._extract_vivado_utilization(content)

        timing_file = vivado_dir / "design_2_timing_summary.rpt"
        if timing_file.exists():
            content = timing_file.read_text()
            timing_data = self._extract_vivado_timing(content)

        return util_data, timing_data

    def _extract_vivado_utilization(self, content: str) -> Dict:
        util = {"utilized": {}, "available": {}, "percent": {}}
        patterns = [
            (r'CLB LUT\s+\|\s*(\d+)\s+\|\s*(\d+)\s+\|\s*([\d.]+)', 'LUT'),
            (r'DSP\s+\|\s*(\d+)\s+\|\s*(\d+)\s+\|\s*([\d.]+)', 'DSP'),
            (r'BRAM\s+\|\s*(\d+)\s+\|\s*(\d+)\s+\|\s*([\d.]+)', 'BRAM'),
            (r'FF\s+\|\s*(\d+)\s+\|\s*(\d+)\s+\|\s*([\d.]+)', 'FF'),
        ]

        for pattern, resource in patterns:
            match = re.search(pattern, content)
            if match:
                util["utilized"][resource] = int(match.group(1))
                util["available"][resource] = int(match.group(2))
                util["percent"][resource] = float(match.group(3))

        return util

    def _extract_vivado_timing(self, content: str) -> Dict:
        timing = {"wns": None, "tns": None, "ws": None}
        wns_match = re.search(r'WNS.*?:\s*([\d.]+)', content)
        tns_match = re.search(r'TNS.*?:\s*([\d.]+)', content)
        ws_match = re.search(r'WHS.*?:\s*([\d.]+)', content)

        if wns_match:
            timing["wns"] = float(wns_match.group(1))
        if tns_match:
            timing["tns"] = float(tns_match.group(1))
        if ws_match:
            timing["ws"] = float(ws_match.group(1))

        return timing

    def query_vivado_tcl(self, query: str) -> Optional[str]:
        """Layer 3: Query Vivado via Tcl interface"""
        try:
            result = subprocess.run(
                ["/opt/Xilinx/Vivado/2023.1/bin/vivado", "-mode", "batch", "-source", "/dev/stdin"],
                input=f"puts [{query}]",
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.stdout.strip()
        except Exception as e:
            return f"Tcl query failed: {e}"

    def get_detailed_utilization(self) -> Dict:
        """Layer 3: Get detailed resource breakdown via Tcl"""
        tcl_queries = {
            "fabric_util": "get_property PART [current_project]",
            "resource_summary": "report_utilization -file /dev/stdout -format json"
        }

        details = {}
        for name, query in tcl_queries.items():
            result = self.query_vivado_tcl(query)
            if result and not result.startswith("Tcl query failed"):
                details[name] = result

        return details

    def generate_recommendations(self) -> List[Dict]:
        """Generate optimization recommendations based on metrics"""
        recommendations = []

        hls = self.results.get("hls", {})
        vivado_util = self.results.get("vivado_util", {})
        vivado_timing = self.results.get("vivado_timing", {})

        dsp_pct = hls.get("DSP_percent", vivado_util.get("percent", {}).get("DSP", 0))
        if dsp_pct > 80:
            recommendations.append({
                "type": "DSP",
                "severity": "warning",
                "issue": f"DSP utilization at {dsp_pct:.1f}% (threshold: 80%)",
                "action": "Reduce unroll factor in HLS pragma or use resource sharing"
            })

        lut_pct = hls.get("LUT_percent", vivado_util.get("percent", {}).get("LUT", 0))
        if lut_pct > 85:
            recommendations.append({
                "type": "LUT",
                "severity": "warning",
                "issue": f"LUT utilization at {lut_pct:.1f}% (threshold: 85%)",
                "action": "Enable logic optimization or reduce unroll factor"
            })

        wns = vivado_timing.get("wns")
        if wns is not None and wns < 0:
            recommendations.append({
                "type": "timing",
                "severity": "critical",
                "issue": f"Negative WNS: {wns} ns",
                "action": "Increase clock period constraint or apply register balancing"
            })

        latency = hls.get("latency_cycles")
        if latency and latency > 10000:
            recommendations.append({
                "type": "latency",
                "severity": "info",
                "issue": f"High latency: {latency} cycles",
                "action": "Add PIPELINE pragma or increase unroll factor for parallel processing"
            })

        bram_pct = hls.get("BRAM_percent", vivado_util.get("percent", {}).get("BRAM", 0))
        if bram_pct > 70:
            recommendations.append({
                "type": "BRAM",
                "severity": "warning",
                "issue": f"BRAM utilization at {bram_pct:.1f}% (threshold: 70%)",
                "action": "Use memory partitioning or reduce buffer sizes"
            })

        if not recommendations:
            recommendations.append({
                "type": "status",
                "severity": "success",
                "issue": "All metrics within acceptable ranges",
                "action": "Design is meeting constraints"
            })

        return recommendations

    def parse_all(self) -> Dict:
        """Main entry point: parse all feedback layers"""
        print("Parsing FPGA design feedback...")

        self.results["timestamp"] = subprocess.run(
            ["date", "+%Y-%m-%d %H:%M:%S"],
            capture_output=True,
            text=True
        ).stdout.strip()

        print("  [Layer 1] Capturing console output...")
        console_errors = self._capture_console_errors()

        print("  [Layer 2] Parsing report files...")
        self.results["hls"] = self.parse_hls_reports()
        vivado_util, vivado_timing = self.parse_vivado_reports()
        self.results["vivado_util"] = vivado_util
        self.results["vivado_timing"] = vivado_timing

        print("  [Layer 3] Querying Tcl interface...")
        self.results["tcl_details"] = self.get_detailed_utilization()

        self.results["recommendations"] = self.generate_recommendations()
        self.results["console_errors"] = console_errors

        return self.results

    def _capture_console_errors(self) -> List[str]:
        errors = []
        log_dir = self.log_dir if self.log_dir.exists() else WORKSPACE / "logs"
        if log_dir.exists():
            for log_file in sorted(log_dir.glob("iteration_*.log"))[-3:]:
                content = log_file.read_text()
                for line in content.splitlines():
                    if re.search(r'ERROR|CRITICAL|FAIL', line, re.IGNORECASE):
                        errors.append(f"{log_file.name}: {line.strip()}")

        return errors[-5:]

    def print_summary(self):
        """Print human-readable summary"""
        print("\n" + "="*60)
        print("FPGA Design Feedback Summary")
        print("="*60)

        hls = self.results.get("hls", {})
        print(f"\nHLS Synthesis:")
        print(f"  Latency: {hls.get('latency_cycles', 'N/A')} cycles")
        print(f"  DSP: {hls.get('DSP_used', 'N/A')} ({hls.get('DSP_percent', 0):.1f}%)")
        print(f"  LUT: {hls.get('LUT_used', 'N/A')} ({hls.get('LUT_percent', 0):.1f}%)")
        print(f"  BRAM: {hls.get('BRAM_used', 'N/A')} ({hls.get('BRAM_percent', 0):.1f}%)")

        vivado_util = self.results.get("vivado_util", {})
        print(f"\nVivado Utilization (Routed):")
        for res in ["LUT", "DSP", "BRAM", "FF"]:
            used = vivado_util.get("utilized", {}).get(res, "N/A")
            pct = vivado_util.get("percent", {}).get(res, 0)
            print(f"  {res}: {used} ({pct:.1f}%)")

        timing = self.results.get("vivado_timing", {})
        print(f"\nTiming:")
        print(f"  WNS: {timing.get('wns', 'N/A')} ns")
        print(f"  TNS: {timing.get('tns', 'N/A')} ns")

        print(f"\nRecommendations ({len(self.results.get('recommendations', []))}):")
        for rec in self.results.get("recommendations", []):
            severity_icon = {"critical": "🔴", "warning": "⚠️", "info": "ℹ️", "success": "✅"}.get(rec.get("severity"), "•")
            print(f"  {severity_icon} [{rec.get('type', 'general').upper()}] {rec.get('issue', '')}")
            print(f"     → {rec.get('action', '')}")

        print("\n" + "="*60)

def main():
    log_dir = sys.argv[1] if len(sys.argv) > 1 else WORKSPACE / "logs"

    parser = FeedbackParser(Path(log_dir))
    results = parser.parse_all()

    parser.print_summary()

    output_file = WORKSPACE / "logs" / "feedback_latest.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(results, indent=2))
    print(f"\nStructured output saved to: {output_file}")

    return 0

if __name__ == "__main__":
    sys.exit(main())