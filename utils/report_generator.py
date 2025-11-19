"""
Report Generator Module
Generates automated analysis reports from Excel data and document summaries
"""

import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import json

class ReportGenerator:
    """Generate automated reports from analysis data"""
    
    def __init__(self, llm_handler, report_dir: Path):
        """
        Initialize Report Generator
        
        Args:
            llm_handler: LLM handler for generating insights
            report_dir: Directory to save reports
        """
        self.llm_handler = llm_handler
        self.report_dir = report_dir
        self.report_dir.mkdir(exist_ok=True)
    
    def generate_excel_report(self, 
                            df: pd.DataFrame, 
                            filename: str,
                            report_type: str = "executive") -> tuple[str, str]:
        """
        Generate report from Excel data analysis
        
        Args:
            df: Pandas DataFrame
            filename: Original file name
            report_type: Type of report (executive, technical, business)
            
        Returns:
            Tuple of (report_content, report_path)
        """
        # Generate data statistics
        stats = self._generate_data_stats(df)
        
        # Generate AI insights
        insights = self._generate_ai_insights(df, stats, report_type)
        
        # Create report based on template
        if report_type == "executive":
            report_content = self._create_executive_report(filename, stats, insights, df)
        elif report_type == "technical":
            report_content = self._create_technical_report(filename, stats, insights, df)
        else:  # business
            report_content = self._create_business_report(filename, stats, insights, df)
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"{report_type}_report_{timestamp}.md"
        report_path = self.report_dir / report_filename
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        return report_content, str(report_path)
    
    def generate_document_report(self, 
                                summary: str, 
                                filename: str,
                                report_type: str = "executive") -> tuple[str, str]:
        """
        Generate report from document summary
        
        Args:
            summary: Document summary text
            filename: Original document name
            report_type: Type of report
            
        Returns:
            Tuple of (report_content, report_path)
        """
        # Generate AI analysis of summary
        insights = self._generate_summary_insights(summary, report_type)
        
        # Create report
        report_content = self._create_document_report(filename, summary, insights, report_type)
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"doc_{report_type}_report_{timestamp}.md"
        report_path = self.report_dir / report_filename
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        return report_content, str(report_path)
    
    def _generate_data_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive statistics from DataFrame"""
        stats = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "columns": list(df.columns),
            "missing_values": df.isnull().sum().to_dict(),
            "data_types": df.dtypes.astype(str).to_dict(),
            "numeric_columns": df.select_dtypes(include=['number']).columns.tolist(),
            "categorical_columns": df.select_dtypes(include=['object']).columns.tolist(),
        }
        
        # Add numeric statistics
        if stats["numeric_columns"]:
            numeric_stats = {}
            for col in stats["numeric_columns"]:
                numeric_stats[col] = {
                    "mean": float(df[col].mean()) if not df[col].isnull().all() else 0,
                    "median": float(df[col].median()) if not df[col].isnull().all() else 0,
                    "min": float(df[col].min()) if not df[col].isnull().all() else 0,
                    "max": float(df[col].max()) if not df[col].isnull().all() else 0,
                    "std": float(df[col].std()) if not df[col].isnull().all() else 0,
                }
            stats["numeric_statistics"] = numeric_stats
        
        return stats
    
    def _generate_ai_insights(self, df: pd.DataFrame, stats: Dict, report_type: str) -> str:
        """Generate AI-powered insights from data"""
        # Create context for LLM
        context = f"""
Analyze this dataset and provide {report_type} insights:

Dataset Overview:
- Total Rows: {stats['total_rows']}
- Total Columns: {stats['total_columns']}
- Columns: {', '.join(stats['columns'][:10])}

Numeric Columns: {', '.join(stats['numeric_columns'][:5]) if stats['numeric_columns'] else 'None'}
Categorical Columns: {', '.join(stats['categorical_columns'][:5]) if stats['categorical_columns'] else 'None'}

Sample Data (first 3 rows):
{df.head(3).to_string()}

Provide 3-5 key insights about this data in a {report_type} style.
"""
        
        try:
            insights = self.llm_handler.generate_response(context)
            return insights
        except Exception as e:
            return f"Unable to generate AI insights: {str(e)}"
    
    def _generate_summary_insights(self, summary: str, report_type: str) -> str:
        """Generate insights from document summary"""
        prompt = f"""
Based on this document summary, provide {report_type} analysis with key takeaways:

Summary:
{summary[:1000]}

Provide 3-5 key points in {report_type} format.
"""
        try:
            insights = self.llm_handler.generate_response(prompt)
            return insights
        except Exception as e:
            return f"Unable to generate insights: {str(e)}"
    
    def _create_executive_report(self, filename: str, stats: Dict, insights: str, df: pd.DataFrame) -> str:
        """Create executive summary report"""
        report = f"""# 📊 Executive Summary Report

**Generated:** {datetime.now().strftime("%B %d, %Y at %I:%M %p")}  
**Source File:** {filename}

---

## 🎯 Key Highlights

{insights}

---

## 📈 Data Overview

| Metric | Value |
|--------|-------|
| Total Records | {stats['total_rows']:,} |
| Data Points | {stats['total_columns']} |
| Data Quality | {100 - (sum(stats['missing_values'].values()) / (stats['total_rows'] * stats['total_columns']) * 100) if stats['total_rows'] * stats['total_columns'] > 0 else 100:.1f}% Complete |

---

## 🔍 Quick Statistics

"""
        # Add numeric statistics
        if "numeric_statistics" in stats:
            report += "### Top Numeric Metrics\n\n"
            for col, values in list(stats["numeric_statistics"].items())[:5]:
                report += f"**{col}:**\n"
                report += f"- Average: {values['mean']:.2f}\n"
                report += f"- Range: {values['min']:.2f} to {values['max']:.2f}\n\n"
        
        # Add data sample using simple string formatting
        report += "\n---\n\n## 📋 Data Sample\n\n"
        report += df.head(5).to_string()
        
        report += """

---

*Report automatically generated by RAG AI Analytic Studio*
"""
        return report
    
    def _create_technical_report(self, filename: str, stats: Dict, insights: str, df: pd.DataFrame) -> str:
        """Create technical analysis report"""
        report = f"""# 🔬 Technical Analysis Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Source:** {filename}  
**Framework:** RAG AI Analytic Studio

---

## 📊 Dataset Specifications


---

## 🔍 AI-Powered Analysis

{insights}

---

## 📈 Column Analysis

### Numeric Columns ({len(stats['numeric_columns'])})
"""
        if "numeric_statistics" in stats:
            report += "\n| Column | Mean | Median | Std Dev | Min | Max |\n"
            report += "|--------|------|--------|---------|-----|-----|\n"
            for col, values in stats["numeric_statistics"].items():
                report += f"| {col} | {values['mean']:.2f} | {values['median']:.2f} | {values['std']:.2f} | {values['min']:.2f} | {values['max']:.2f} |\n"
        
        report += f"""

### Categorical Columns ({len(stats['categorical_columns'])})
{', '.join(stats['categorical_columns'][:10])}

---

## 🧹 Data Quality Assessment

"""
        total_missing = sum(stats['missing_values'].values())
        if total_missing > 0:
            report += "### Missing Values Detected\n\n"
            for col, missing in stats['missing_values'].items():
                if missing > 0:
                    pct = (missing / stats['total_rows']) * 100
                    report += f"- **{col}:** {missing} ({pct:.1f}%)\n"
        else:
            report += "✅ No missing values detected\n"
        
        report += f"""

---

## 📋 Data Preview

{df.head(10).to_string()}

---

*Technical report generated using Llama 3.2 3B via Ollama*
"""
        return report
    
    def _create_business_report(self, filename: str, stats: Dict, insights: str, df: pd.DataFrame) -> str:
        total_cells = stats['total_rows'] * stats['total_columns']
        completeness = 100 - (sum(stats['missing_values'].values()) / total_cells * 100) if total_cells > 0 else 100
        
        report = f"""# 💼 Business Intelligence Report

**Report Date:** {datetime.now().strftime("%B %d, %Y")}  
**Data Source:** {filename}  
**Prepared By:** RAG AI Analytic Studio

---

## 🎯 Executive Summary

This report analyzes **{stats['total_rows']:,} records** across **{stats['total_columns']} key metrics** to provide actionable business insights.

---

## 💡 Key Business Insights

{insights}

---

## 📊 Business Metrics Dashboard

| KPI | Value |
|-----|-------|
| Total Data Records | {stats['total_rows']:,} |
| Metrics Tracked | {stats['total_columns']} |
| Data Completeness | {completeness:.1f}% |
| Numeric Indicators | {len(stats['numeric_columns'])} |

---

## 📈 Performance Metrics

"""
        if "numeric_statistics" in stats:
            report += "### Key Performance Indicators\n\n"
            for col, values in list(stats["numeric_statistics"].items())[:5]:
                report += f"**{col}:**\n"
                report += f"- Avg Performance: {values['mean']:.2f}\n"
                report += f"- Best: {values['max']:.2f} | Worst: {values['min']:.2f}\n"
                report += f"- Variability: {values['std']:.2f}\n\n"
        
        report += f"""
---

## 🔍 Data Snapshot

{df.head(5).to_string()}

---

## 📌 Recommendations

Based on the analysis:
1. Monitor key metrics with high variability
2. Investigate any data quality issues identified
3. Focus on columns showing significant trends
4. Consider deeper analysis on outlier values

---

*Business report powered by AI - RAG AI Analytic Studio*
"""
        return report
    
    def _create_document_report(self, filename: str, summary: str, insights: str, report_type: str) -> str:
        report = f"""# 📄 Document Analysis Report - {report_type.title()}

**Generated:** {datetime.now().strftime("%B %d, %Y at %I:%M %p")}  
**Document:** {filename}  
**Report Type:** {report_type.upper()}

---

## 📋 Document Summary

{summary}

---

## 💡 Key Insights & Analysis

{insights}

---

## 🎯 Analysis Methodology

This report was generated using:
- **AI Model:** Llama 3.2 3B (Ollama)
- **Technique:** RAG (Retrieval-Augmented Generation)
- **Processing:** Local, privacy-focused analysis
- **Report Format:** {report_type.title()}

---

## 📌 Key Takeaways

Based on the document analysis, this report provides:
- Comprehensive summary of document content
- AI-powered insights and interpretations
- Actionable recommendations where applicable
- Context-aware analysis tailored to {report_type} perspective

---

*Document analysis powered by RAG AI Analytic Studio*  
*Report generated on {datetime.now().strftime("%B %d, %Y at %I:%M %p")}*
"""
        return report
