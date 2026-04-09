// PRE Web GUI — Document generation tool
// Creates .txt, .xml, .docx, .xlsx, .pdf files in ~/.pre/artifacts/

const fs = require('fs');
const path = require('path');
const { ARTIFACTS_DIR } = require('../constants');

// Ensure artifacts directory exists
if (!fs.existsSync(ARTIFACTS_DIR)) {
  fs.mkdirSync(ARTIFACTS_DIR, { recursive: true });
}

function slugify(title) {
  return (title || 'document')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '')
    .slice(0, 60);
}

/**
 * Create a document file.
 * @param {Object} args - { title, content, format, sheets? }
 *   format: 'txt' | 'xml' | 'docx' | 'xlsx' | 'pdf'
 *   sheets: for xlsx, array of { name, headers, rows }
 * @returns {string} Result with download path
 */
async function createDocument(args) {
  const title = args.title || 'Untitled Document';
  const content = args.content || '';
  const format = (args.format || 'txt').toLowerCase().replace('.', '');
  const slug = slugify(title);
  const timestamp = Date.now().toString(36);

  let filename, filePath;

  switch (format) {
    case 'txt': {
      filename = `${slug}-${timestamp}.txt`;
      filePath = path.join(ARTIFACTS_DIR, filename);
      fs.writeFileSync(filePath, content);
      break;
    }

    case 'xml': {
      filename = `${slug}-${timestamp}.xml`;
      filePath = path.join(ARTIFACTS_DIR, filename);
      fs.writeFileSync(filePath, content);
      break;
    }

    case 'docx':
    case 'doc': {
      filename = `${slug}-${timestamp}.docx`;
      filePath = path.join(ARTIFACTS_DIR, filename);
      await createDocx(filePath, title, content);
      break;
    }

    case 'xlsx':
    case 'xls': {
      filename = `${slug}-${timestamp}.xlsx`;
      filePath = path.join(ARTIFACTS_DIR, filename);
      await createXlsx(filePath, title, content, args.sheets);
      break;
    }

    case 'pdf': {
      filename = `${slug}-${timestamp}.pdf`;
      filePath = path.join(ARTIFACTS_DIR, filename);
      await createPdf(filePath, title, content);
      break;
    }

    default:
      return `Error: Unsupported format '${format}'. Supported: txt, xml, docx, xlsx, pdf`;
  }

  const size = fs.statSync(filePath).size;
  const sizeStr = size > 1024 ? `${(size / 1024).toFixed(1)}KB` : `${size}B`;
  return `Document created: ${filename} (${sizeStr})\nDownload: /artifacts/${filename}`;
}

/**
 * Create a .docx file from text content
 */
async function createDocx(filePath, title, content) {
  const docx = require('docx');

  const paragraphs = [];

  // Title
  paragraphs.push(new docx.Paragraph({
    children: [new docx.TextRun({ text: title, bold: true, size: 32, font: 'Calibri' })],
    spacing: { after: 300 },
  }));

  // Parse content — handle markdown-like formatting
  const lines = content.split('\n');
  for (const line of lines) {
    if (!line.trim()) {
      paragraphs.push(new docx.Paragraph({ spacing: { after: 100 } }));
      continue;
    }

    // Headings
    const h1Match = line.match(/^#\s+(.+)/);
    const h2Match = line.match(/^##\s+(.+)/);
    const h3Match = line.match(/^###\s+(.+)/);

    if (h1Match) {
      paragraphs.push(new docx.Paragraph({
        children: [new docx.TextRun({ text: h1Match[1], bold: true, size: 28, font: 'Calibri' })],
        spacing: { before: 300, after: 150 },
      }));
    } else if (h2Match) {
      paragraphs.push(new docx.Paragraph({
        children: [new docx.TextRun({ text: h2Match[1], bold: true, size: 24, font: 'Calibri' })],
        spacing: { before: 200, after: 100 },
      }));
    } else if (h3Match) {
      paragraphs.push(new docx.Paragraph({
        children: [new docx.TextRun({ text: h3Match[1], bold: true, size: 22, font: 'Calibri' })],
        spacing: { before: 150, after: 100 },
      }));
    } else if (line.startsWith('- ') || line.startsWith('* ')) {
      // Bullet points
      paragraphs.push(new docx.Paragraph({
        children: [new docx.TextRun({ text: line.slice(2), font: 'Calibri', size: 22 })],
        bullet: { level: 0 },
      }));
    } else {
      // Regular paragraph — handle bold markers
      const runs = [];
      const parts = line.split(/(\*\*[^*]+\*\*)/);
      for (const part of parts) {
        if (part.startsWith('**') && part.endsWith('**')) {
          runs.push(new docx.TextRun({ text: part.slice(2, -2), bold: true, font: 'Calibri', size: 22 }));
        } else {
          runs.push(new docx.TextRun({ text: part, font: 'Calibri', size: 22 }));
        }
      }
      paragraphs.push(new docx.Paragraph({ children: runs, spacing: { after: 80 } }));
    }
  }

  const doc = new docx.Document({
    sections: [{ children: paragraphs }],
  });

  const buffer = await docx.Packer.toBuffer(doc);
  fs.writeFileSync(filePath, buffer);
}

/**
 * Create a .xlsx file from structured data or text
 */
async function createXlsx(filePath, title, content, sheets) {
  const ExcelJS = require('exceljs');
  const workbook = new ExcelJS.Workbook();
  workbook.creator = 'PRE';
  workbook.created = new Date();

  if (sheets && Array.isArray(sheets) && sheets.length > 0) {
    // Structured data: array of { name, headers, rows }
    for (const sheet of sheets) {
      const ws = workbook.addWorksheet(sheet.name || 'Sheet');
      if (sheet.headers) {
        const headerRow = ws.addRow(sheet.headers);
        headerRow.font = { bold: true, size: 11 };
        headerRow.fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FF4472C4' } };
        headerRow.font = { bold: true, color: { argb: 'FFFFFFFF' }, size: 11 };
      }
      if (sheet.rows) {
        for (const row of sheet.rows) {
          ws.addRow(row);
        }
      }
      // Auto-width columns
      ws.columns.forEach(col => {
        let maxLen = 10;
        col.eachCell({ includeEmpty: false }, cell => {
          const len = (cell.value || '').toString().length;
          if (len > maxLen) maxLen = len;
        });
        col.width = Math.min(maxLen + 2, 50);
      });
    }
  } else {
    // Parse text content into a table
    // Try to detect pipe-delimited table or CSV-like content
    const ws = workbook.addWorksheet(title.slice(0, 31));
    const lines = content.split('\n').filter(l => l.trim());

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();
      // Skip markdown separator rows
      if (/^[\|\s:\-]+$/.test(line)) continue;

      let cells;
      if (line.includes('|')) {
        cells = line.replace(/^\||\|$/g, '').split('|').map(c => c.trim());
      } else if (line.includes('\t')) {
        cells = line.split('\t');
      } else if (line.includes(',')) {
        cells = line.split(',').map(c => c.trim().replace(/^"|"$/g, ''));
      } else {
        cells = [line];
      }

      const row = ws.addRow(cells);
      if (i === 0) {
        row.font = { bold: true, color: { argb: 'FFFFFFFF' }, size: 11 };
        row.fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FF4472C4' } };
      }
    }

    ws.columns.forEach(col => {
      let maxLen = 10;
      col.eachCell({ includeEmpty: false }, cell => {
        const len = (cell.value || '').toString().length;
        if (len > maxLen) maxLen = len;
      });
      col.width = Math.min(maxLen + 2, 50);
    });
  }

  await workbook.xlsx.writeFile(filePath);
}

/**
 * Create a .pdf file from text content
 */
async function createPdf(filePath, title, content) {
  const PDFDocument = require('pdfkit');

  return new Promise((resolve, reject) => {
    const doc = new PDFDocument({ margin: 50, size: 'LETTER' });
    const stream = fs.createWriteStream(filePath);
    doc.pipe(stream);

    // Title
    doc.fontSize(20).font('Helvetica-Bold').text(title, { align: 'left' });
    doc.moveDown(0.5);
    doc.moveTo(50, doc.y).lineTo(562, doc.y).stroke('#cccccc');
    doc.moveDown(0.5);

    // Content
    const lines = content.split('\n');
    for (const line of lines) {
      if (!line.trim()) {
        doc.moveDown(0.3);
        continue;
      }

      const h1 = line.match(/^#\s+(.+)/);
      const h2 = line.match(/^##\s+(.+)/);
      const h3 = line.match(/^###\s+(.+)/);

      if (h1) {
        doc.moveDown(0.5);
        doc.fontSize(16).font('Helvetica-Bold').text(h1[1]);
      } else if (h2) {
        doc.moveDown(0.3);
        doc.fontSize(14).font('Helvetica-Bold').text(h2[1]);
      } else if (h3) {
        doc.moveDown(0.2);
        doc.fontSize(12).font('Helvetica-Bold').text(h3[1]);
      } else if (line.startsWith('- ') || line.startsWith('* ')) {
        doc.fontSize(11).font('Helvetica').text(`  •  ${line.slice(2)}`, { indent: 15 });
      } else {
        // Handle inline bold
        const stripped = line.replace(/\*\*([^*]+)\*\*/g, '$1');
        doc.fontSize(11).font('Helvetica').text(stripped);
      }
    }

    doc.end();
    stream.on('finish', resolve);
    stream.on('error', reject);
  });
}

module.exports = { createDocument };
