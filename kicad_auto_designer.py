#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KiCad Auto Designer - Automatic PCB Component Layout Plugin
For KiCad 9.0

This plugin provides automatic optimized layout functionality for PCB components.
"""

import os
import pcbnew
import wx
import math
import random
import logging

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[AUTO DESIGNER] %(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class KiCadAutoDesigner:
    """
    Main plugin class for automatic PCB component layout.
    
    This class handles:
    - Communication with KiCad's PCB editor
    - Optimized component placement algorithm
    - Design rule checking
    """
    
    def __init__(self):
        """Initialize the KiCad Auto Designer plugin."""
        self.board = None
        self.logger = logger
        
    def Initialize(self):
        """
        Initialize the plugin when KiCad loads it.
        Called automatically by KiCad.
        """
        self.logger.info("KiCad Auto Designer initialized")
        
    def Show(self):
        """
        Show the plugin information.
        Called when KiCad queries plugin details.
        """
        return "KiCad Auto Designer - Automatic PCB Component Layout"
    
    def get_footprints(self):
        """Get all footprints from the board."""
        if not self.board:
            return []
        return self.board.GetFootprints()
    
    def get_board_bounds(self):
        """Get the board boundaries from edge cuts."""
        edge_cuts = []
        for drawing in self.board.GetDrawings():
            if drawing.GetLayer() == pcbnew.Edge_Cuts:
                edge_cuts.append(drawing)
        
        if not edge_cuts:
            # No edge cut found, return default bounds
            self.logger.warning("No edge cut found, using default board size")
            return (0, 0, int(200 * 1_000_000), int(150 * 1_000_000))  # Default 200x150mm
        
        # Get bounding box of all edge cuts
        min_x = float('inf')
        min_y = float('inf')
        max_x = float('-inf')
        max_y = float('-inf')
        
        for drawing in edge_cuts:
            bbox = drawing.GetBoundingBox()
            min_x = min(min_x, bbox.GetLeft())
            min_y = min(min_y, bbox.GetTop())
            max_x = max(max_x, bbox.GetRight())
            max_y = max(max_y, bbox.GetBottom())
        
        return (int(min_x), int(min_y), int(max_x), int(max_y))
    
    def is_connector(self, footprint):
        """Check if footprint is a connector."""
        ref = footprint.GetReference().upper()
        connector_keywords = ['USB', 'CONN', 'J', 'RJ', 'HDMI', 'ETHERNET', 'JACK', 'PORT', 'PLUG']
        return any(keyword in ref for keyword in connector_keywords)
    
    def is_antenna_component(self, footprint):
        """Check if footprint is an antenna component."""
        ref = footprint.GetReference().upper()
        antenna_keywords = ['ANTENNA', 'ANT', 'U2']
        return any(keyword in ref for keyword in antenna_keywords)
    
    def get_connector_type(self, footprint):
        """Determine connector type and optimal edge placement."""
        ref = footprint.GetReference().upper()
        if 'BARREL' in ref or 'JACK' in ref:
            return 'barrel'
        elif 'USB' in ref:
            return 'usb'
        elif 'RJ' in ref or 'ETHERNET' in ref:
            return 'ethernet'
        else:
            return 'generic'
    
    def layout_optimized(self):
        """
        Connectivity-aware professional layout: grouped by signal flow.
        - Wireless subsystem (antenna, U1, related caps) - isolated
        - Power management (AP62, regulators, filtering caps)
        - Main processor and support (U4, bypass caps)
        - Motor control circuit (DRV87, motor connector)
        - Interfaces (USB, UART, JTAG) and remaining passives
        - All packed tightly for compact, professional layout
        """
        footprints = self.get_footprints()
        
        if not footprints:
            self.logger.warning("No footprints found on board")
            return False
        
        min_spacing_nm = int(0.01 * 1_000_000)
        min_x, min_y, max_x, max_y = self.get_board_bounds()
        margin_nm = int(1 * 1_000_000)
        inner_min_x = min_x + margin_nm
        inner_min_y = min_y + margin_nm
        inner_max_x = max_x - margin_nm
        inner_max_y = max_y - margin_nm
        
        # Group by connectivity and signal flow
        
        # 1. Wireless subsystem (U1, antenna, RF-related caps)
        wireless = [fp for fp in footprints if any(x in fp.GetReference().upper() for x in ['U1', 'ANTENNA', 'ANT', 'U2'])]
        wireless_caps = [fp for fp in footprints if any(x in fp.GetReference().upper() for x in ['C1', 'C2', 'C4', 'C5'])]
        wireless_group = wireless + wireless_caps
        
        # 2. Power management (AP62, regulators, big filter caps)
        power_management = [fp for fp in footprints if any(x in fp.GetReference().upper() for x in ['AP62', 'AP6225', 'D12V'])]
        power_caps = [fp for fp in footprints if any(x in fp.GetReference().upper() for x in ['C3', 'C8', 'C_bulk'])]
        power_group = power_management + power_caps
        
        # 3. Main processor (U4, its bypass caps)
        main_processor = [fp for fp in footprints if any(x in fp.GetReference().upper() for x in ['U4'])]
        processor_caps = [fp for fp in footprints if any(x in fp.GetReference().upper() for x in ['C6', 'C7'])]
        processor_group = main_processor + processor_caps
        
        # 4. Motor control circuit (DRV87, motor connector, support)
        motor_group = [fp for fp in footprints if any(x in fp.GetReference().upper() for x in ['DRV', 'MOTOR', 'U3', 'R8', 'R9', 'R10'])]
        
        # 5. Interfaces (USB, UART, JTAG connectors)
        interfaces = [fp for fp in footprints if self.is_connector(fp)]
        
        # 6. Remaining passives not yet categorized
        all_categorized = wireless_group + power_group + processor_group + motor_group + interfaces
        passives = [fp for fp in footprints if fp not in all_categorized]
        
        self.logger.info(f"Connectivity-based layout: wireless({len(wireless_group)}), power({len(power_group)}), "
                        f"processor({len(processor_group)}), motor({len(motor_group)}), interfaces({len(interfaces)}), passives({len(passives)})")
        
        # Order by signal flow: Power first (foundation), then wireless, processor, motor, interfaces, remaining
        # This keeps electrically related components together
        ordered_components = power_group + wireless_group + processor_group + motor_group + interfaces + passives
        
        # Pack all components together (ordered for natural grouping by connectivity)
        self._place_components_in_tight_grid_in_area(ordered_components, min_spacing_nm, inner_min_x, inner_min_y, inner_max_x, inner_max_y)

        # Run 2D compaction to optimize and tighten
        try:
            if ordered_components:
                self._compact_2d_pack(ordered_components, min_spacing_nm, inner_min_x, inner_min_y, inner_max_x, inner_max_y)
        except Exception as e:
            self.logger.warning(f"2D compaction pass failed: {e}")

        return True
    
    def _place_components_in_tight_grid(self, footprints, min_spacing_nm):
        """Place components in the tightest possible grid packing."""
        if not footprints:
            return
        
        # Calculate each component's size
        components_with_size = []
        for fp in footprints:
            bbox = fp.GetBoundingBox()
            width = bbox.GetWidth()
            height = bbox.GetHeight()
            components_with_size.append({
                'fp': fp,
                'width': width,
                'height': height,
                'size': max(width, height)
            })
        
        # Sort by size (largest first) for better packing
        components_with_size.sort(key=lambda x: -x['size'])
        
        # Calculate optimal grid - aim for roughly square layout
        n = len(components_with_size)
        cols = max(1, int(math.sqrt(n)))
        
        # Place components starting from origin
        x_pos = 0
        y_pos = 0
        max_y_in_row = 0
        
        for idx, comp_data in enumerate(components_with_size):
            fp = comp_data['fp']
            width = comp_data['width']
            height = comp_data['height']
            
            # Position component at center of its cell
            fp.SetPosition(pcbnew.VECTOR2I(int(x_pos + width // 2), int(y_pos + height // 2)))
            
            # Track max height in row
            max_y_in_row = max(max_y_in_row, height)
            
            # Move to next position
            x_pos += width + min_spacing_nm
            
            # Check if we should move to next row
            if (idx + 1) % cols == 0:
                y_pos += max_y_in_row + min_spacing_nm
                x_pos = 0
                max_y_in_row = 0
            
            self.logger.info(f"Placed {fp.GetReference()} in grid")
        
        return True

    def _place_components_in_tight_grid_in_area(self, footprints, spacing_nm, area_min_x, area_min_y, area_max_x, area_max_y):
        """Pack components tightly into the specified rectangular area using improved shelf packing.
        
        Targets a roughly square layout by calculating optimal row width based on total area.
        """
        if not footprints:
            return True
        
        # Prepare components with sizes
        comps = []
        total_area = 0
        for fp in footprints:
            bbox = fp.GetBoundingBox()
            w = bbox.GetWidth()
            h = bbox.GetHeight()
            total_area += w * h
            comps.append({'fp': fp, 'w': w, 'h': h, 'rot': False})

        # Sort by height (tallest first) - NFDH heuristic
        comps.sort(key=lambda c: -c['h'])

        area_width = area_max_x - area_min_x
        area_height = area_max_y - area_min_y

        # Target a more square layout by calculating target width from total area
        target_aspect = 1.0  # aim for square
        target_width = int(math.sqrt(total_area / target_aspect))
        # Clamp to available width
        target_width = max(int(area_width * 0.3), min(target_width, area_width))
        
        rows = []  # each row: {'y': y, 'height': h, 'items': [placed_items], 'used_w': used_width}

        # Place items into rows, trying to fill to target_width
        for c in comps:
            placed_flag = False
            w = c['w']
            h = c['h']
            
            # Try existing rows
            for row in rows:
                remaining_w = target_width - row['used_w'] - spacing_nm
                # Try without rotation
                if w <= remaining_w:
                    x = int(area_min_x + row['used_w'] + spacing_nm)
                    y = int(area_min_y + row['y'])
                    row['items'].append({'fp': c['fp'], 'x': x, 'y': y, 'w': w, 'h': h, 'rot': False})
                    row['used_w'] += w + spacing_nm
                    row['height'] = max(row['height'], h)
                    placed_flag = True
                    break
                # Try rotated placement
                if h <= remaining_w:
                    x = int(area_min_x + row['used_w'] + spacing_nm)
                    y = int(area_min_y + row['y'])
                    row['items'].append({'fp': c['fp'], 'x': x, 'y': y, 'w': h, 'h': w, 'rot': True})
                    row['used_w'] += h + spacing_nm
                    row['height'] = max(row['height'], w)
                    placed_flag = True
                    break

            if placed_flag:
                continue

            # Create new row if not placed
            new_row_y = 0 if not rows else rows[-1]['y'] + rows[-1]['height'] + spacing_nm
            if new_row_y + h > area_height:
                self.logger.warning("Inner area vertical overflow during packing")

            # Decide if rotate on new row
            if w <= target_width - spacing_nm:
                item_w, item_h, rot_flag = w, h, False
            elif h <= target_width - spacing_nm:
                item_w, item_h, rot_flag = h, w, True
            else:
                item_w, item_h, rot_flag = w, h, False

            row = {'y': new_row_y, 'height': item_h, 'items': [], 'used_w': 0}
            x = int(area_min_x + spacing_nm)
            y = int(area_min_y + row['y'])
            row['items'].append({'fp': c['fp'], 'x': x, 'y': y, 'w': item_w, 'h': item_h, 'rot': rot_flag})
            row['used_w'] = item_w + spacing_nm
            rows.append(row)

        # Compact rows: shift items left and pack rows vertically
        for row in rows:
            row['items'].sort(key=lambda it: it['x'])
            cur_x = int(area_min_x + spacing_nm)
            for it in row['items']:
                it['x'] = cur_x
                cur_x += it['w'] + spacing_nm
            row['used_w'] = cur_x - int(area_min_x)

        # Compute actual y positions for rows
        cur_y = 0
        for row in rows:
            row['y'] = cur_y
            cur_y += row['height'] + spacing_nm

        # Apply placements
        placed_count = 0
        for row in rows:
            for it in row['items']:
                fp = it['fp']
                cx = int(it['x'] + it['w'] // 2)
                cy = int(area_min_y + row['y'] + it['h'] // 2)
                fp.SetPosition(pcbnew.VECTOR2I(cx, cy))
                if it.get('rot'):
                    try:
                        fp.SetOrientation(pcbnew.EDA_ANGLE(90, pcbnew.DEGREES_T))
                    except Exception:
                        pass
                placed_count += 1

        self.logger.info(f"Packed {placed_count} components into inner area using improved shelf packing ({len(rows)} rows, target_width={target_width/1_000_000:.2f}mm)")
        return True

    def _compact_2d_pack(self, footprints, spacing_nm, area_min_x, area_min_y, area_max_x, area_max_y, max_iter=80):
        """Greedy 2D compaction with local moves and gravity.

        Tries to reduce overlaps first, then compresses the occupied area. All
        work is in-memory and applied back to footprints at the end.
        """
        if not footprints:
            return True

        comps = []
        for fp in footprints:
            bbox = fp.GetBoundingBox()
            cx = int((bbox.GetLeft() + bbox.GetRight()) / 2)
            cy = int((bbox.GetTop() + bbox.GetBottom()) / 2)
            w = bbox.GetWidth()
            h = bbox.GetHeight()
            comps.append({'fp': fp, 'cx': cx, 'cy': cy, 'w': w, 'h': h, 'rot': False})

        def overlap_area(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2):
            ix = max(0, min(ax2, bx2) - max(ax1, bx1))
            iy = max(0, min(ay2, by2) - max(ay1, by1))
            return ix * iy

        def rectangles_for_comp(comp):
            half_w = comp['w'] // 2
            half_h = comp['h'] // 2
            return (
                comp['cx'] - half_w - spacing_nm,
                comp['cy'] - half_h - spacing_nm,
                comp['cx'] + half_w + spacing_nm,
                comp['cy'] + half_h + spacing_nm,
            )

        def total_overlap(comps_list):
            total = 0
            for i in range(len(comps_list)):
                ax1, ay1, ax2, ay2 = rectangles_for_comp(comps_list[i])
                for j in range(i + 1, len(comps_list)):
                    bx1, by1, bx2, by2 = rectangles_for_comp(comps_list[j])
                    total += overlap_area(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2)
            return total

        def bbox_area(comps_list):
            min_x = min(c['cx'] - c['w'] // 2 - spacing_nm for c in comps_list)
            max_x = max(c['cx'] + c['w'] // 2 + spacing_nm for c in comps_list)
            min_y = min(c['cy'] - c['h'] // 2 - spacing_nm for c in comps_list)
            max_y = max(c['cy'] + c['h'] // 2 + spacing_nm for c in comps_list)
            return (max_x - min_x) * (max_y - min_y)

        def score(comps_list, area_weight=0.00001):
            return total_overlap(comps_list) + area_weight * bbox_area(comps_list)

        def total_overlap_for_candidate(idx, cand_x, cand_y, cand_w, cand_h):
            ax1 = cand_x - cand_w // 2 - spacing_nm
            ay1 = cand_y - cand_h // 2 - spacing_nm
            ax2 = cand_x + cand_w // 2 + spacing_nm
            ay2 = cand_y + cand_h // 2 + spacing_nm
            total = 0
            for j, c2 in enumerate(comps):
                if j == idx:
                    continue
                bx = c2['cx']
                by = c2['cy']
                bw = c2['w']
                bh = c2['h']
                bx1 = bx - bw // 2 - spacing_nm
                by1 = by - bh // 2 - spacing_nm
                bx2 = bx + bw // 2 + spacing_nm
                by2 = by + bh // 2 + spacing_nm
                total += overlap_area(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2)
            return total

        def clamp_center(x, y, half_w, half_h):
            min_cx = area_min_x + half_w + spacing_nm
            max_cx = area_max_x - half_w - spacing_nm
            min_cy = area_min_y + half_h + spacing_nm
            max_cy = area_max_y - half_h - spacing_nm
            return int(max(min_cx, min(max_cx, x))), int(max(min_cy, min(max_cy, y)))

        def apply_gravity(step):
            moved = False
            for c in comps:
                for dx, dy in ((-step, 0), (0, -step)):
                    nx = c['cx'] + dx
                    ny = c['cy'] + dy
                    nx, ny = clamp_center(nx, ny, c['w'] // 2, c['h'] // 2)
                    before = total_overlap_for_candidate(comps.index(c), c['cx'], c['cy'], c['w'], c['h'])
                    after = total_overlap_for_candidate(comps.index(c), nx, ny, c['w'], c['h'])
                    if after <= before:
                        c['cx'] = nx
                        c['cy'] = ny
                        moved = moved or (after < before)
            return moved

        def attempt_best_swap():
            best = None
            base_score = score(comps)
            max_checks = min(30, len(comps) * 2)
            indices = list(range(len(comps)))
            random.shuffle(indices)
            checks = 0
            for i in indices:
                for j in indices:
                    if i >= j:
                        continue
                    checks += 1
                    if checks > max_checks:
                        return False
                    ci = comps[i]
                    cj = comps[j]
                    ci_pos = (ci['cx'], ci['cy'])
                    cj_pos = (cj['cx'], cj['cy'])
                    ci_rot = ci['rot']
                    cj_rot = cj['rot']
                    ci['cx'], cj['cx'] = cj_pos[0], ci_pos[0]
                    ci['cy'], cj['cy'] = cj_pos[1], ci_pos[1]
                    ci['rot'], cj['rot'] = cj_rot, ci_rot
                    new_score = score(comps)
                    ci['cx'], cj['cx'] = ci_pos[0], cj_pos[0]
                    ci['cy'], cj['cy'] = ci_pos[1], cj_pos[1]
                    ci['rot'], cj['rot'] = ci_rot, cj_rot
                    if new_score < base_score:
                        best = (i, j)
                        base_score = new_score
            if best is None:
                return False
            i, j = best
            comps[i]['cx'], comps[j]['cx'] = comps[j]['cx'], comps[i]['cx']
            comps[i]['cy'], comps[j]['cy'] = comps[j]['cy'], comps[i]['cy']
            comps[i]['rot'], comps[j]['rot'] = comps[j]['rot'], comps[i]['rot']
            return True

        step = max(spacing_nm, int(0.05 * 1_000_000))

        for it in range(max_iter):
            improved = False
            order = sorted(range(len(comps)), key=lambda i: -(comps[i]['w'] * comps[i]['h']))
            for idx in order:
                c = comps[idx]
                best_x = c['cx']
                best_y = c['cy']
                best_w = c['w']
                best_h = c['h']
                best_rot = c['rot']
                best_score = total_overlap_for_candidate(idx, best_x, best_y, best_w, best_h)

                for dx in (-step, 0, step):
                    for dy in (-step, 0, step):
                        for rot in (False, True):
                            tw = c['w']
                            th = c['h']
                            if rot:
                                tw, th = th, tw
                            nx = c['cx'] + dx
                            ny = c['cy'] + dy
                            nx, ny = clamp_center(nx, ny, tw // 2, th // 2)
                            score_local = total_overlap_for_candidate(idx, nx, ny, tw, th)
                            if score_local < best_score:
                                best_score = score_local
                                best_x = nx
                                best_y = ny
                                best_w = tw
                                best_h = th
                                best_rot = rot

                if best_score < total_overlap_for_candidate(idx, c['cx'], c['cy'], c['w'], c['h']):
                    c['cx'] = best_x
                    c['cy'] = best_y
                    c['w'] = best_w
                    c['h'] = best_h
                    c['rot'] = best_rot
                    improved = True

            swap_done = attempt_best_swap()
            gravity_done = apply_gravity(step)
            improved = improved or swap_done or gravity_done
            if not improved:
                break

        for c in comps:
            fp = c['fp']
            try:
                if c['rot']:
                    fp.SetOrientation(pcbnew.EDA_ANGLE(90, pcbnew.DEGREES_T))
                else:
                    fp.SetOrientation(pcbnew.EDA_ANGLE(0, pcbnew.DEGREES_T))
            except Exception:
                pass
            fp.SetPosition(pcbnew.VECTOR2I(int(c['cx']), int(c['cy'])))

        self.logger.info(f"2D compaction finished after {it+1} iterations")
        return True

    def _move_connectors_to_edges(self):
        """Move all connectors to board edges."""
        footprints = self.get_footprints()
        connectors = [fp for fp in footprints if self.is_connector(fp)]
        
        if not connectors:
            return
        
        # Get current layout bounds
        min_x = float('inf')
        min_y = float('inf')
        max_x = float('-inf')
        max_y = float('-inf')
        
        for fp in footprints:
            bbox = fp.GetBoundingBox()
            min_x = min(min_x, bbox.GetLeft())
            min_y = min(min_y, bbox.GetTop())
            max_x = max(max_x, bbox.GetRight())
            max_y = max(max_y, bbox.GetBottom())
        
        margin_nm = int(2 * 1_000_000)  # 2mm margin reserved for connectors (pull edges closer)
        edge_spacing_nm = int(2 * 1_000_000)  # 2mm between edge components to prevent overlap
        
        # Sort connectors by size (place larger first for stability)
        sorted_fps = sorted(connectors, key=lambda fp: -max(fp.GetBoundingBox().GetWidth(), fp.GetBoundingBox().GetHeight()))

        # Fill left and right edges; avoid top (where main cluster is dense)
        available_side = (max_y - margin_nm) - (min_y + margin_nm)
        run_left = 0
        run_right = 0
        left_fps = []
        right_fps = []
        top_fps = []
        bottom_fps = []
        
        for fp in sorted_fps:
            h = fp.GetBoundingBox().GetHeight()  # height when rotated 90°
            add_len = h if not left_fps else h + edge_spacing_nm
            if run_left + add_len <= available_side:
                left_fps.append(fp)
                run_left += add_len
            else:
                add_len_r = h if not right_fps else h + edge_spacing_nm
                if run_right + add_len_r <= available_side:
                    right_fps.append(fp)
                    run_right += add_len_r
                else:
                    top_fps.append(fp)
        
        def enforce_horizontal_spacing(fp_list, x_min, x_max):
            if not fp_list:
                return
            fp_list.sort(key=lambda f: f.GetBoundingBox().GetLeft())
            total_w = sum(f.GetBoundingBox().GetWidth() for f in fp_list)
            gaps = max(1, len(fp_list) - 1)
            available = (x_max - x_min) - total_w
            spacing = edge_spacing_nm if available <= 0 else max(edge_spacing_nm, available // gaps)
            cur_left = x_min
            positions = []
            for fp in fp_list:
                bb = fp.GetBoundingBox()
                w = bb.GetWidth()
                cx = cur_left + w // 2
                positions.append((fp, cx))
                cur_left = cur_left + w + spacing
            # Clamp run inside [x_min, x_max]
            if positions:
                first_left = positions[0][1] - fp_list[0].GetBoundingBox().GetWidth() // 2
                last_right = positions[-1][1] + fp_list[-1].GetBoundingBox().GetWidth() // 2
                shift = 0
                if last_right > x_max:
                    shift = x_max - last_right
                if first_left + shift < x_min:
                    shift = x_min - first_left
                for fp, cx in positions:
                    pos = fp.GetPosition()
                    fp.SetPosition(pcbnew.VECTOR2I(int(cx + shift), pos.y))

        def enforce_vertical_spacing(fp_list, y_min, y_max):
            if not fp_list:
                return
            fp_list.sort(key=lambda f: f.GetBoundingBox().GetTop())
            total_h = sum(f.GetBoundingBox().GetHeight() for f in fp_list)
            gaps = max(1, len(fp_list) - 1)
            available = (y_max - y_min) - total_h
            spacing = edge_spacing_nm if available <= 0 else max(edge_spacing_nm, available // gaps)
            cur_top = y_min
            positions = []
            for fp in fp_list:
                bb = fp.GetBoundingBox()
                h = bb.GetHeight()
                cy = cur_top + h // 2
                positions.append((fp, cy))
                cur_top = cur_top + h + spacing
            if positions:
                first_top = positions[0][1] - fp_list[0].GetBoundingBox().GetHeight() // 2
                last_bottom = positions[-1][1] + fp_list[-1].GetBoundingBox().GetHeight() // 2
                shift = 0
                if last_bottom > y_max:
                    shift = y_max - last_bottom
                if first_top + shift < y_min:
                    shift = y_min - first_top
                for fp, cy in positions:
                    pos = fp.GetPosition()
                    fp.SetPosition(pcbnew.VECTOR2I(pos.x, int(cy + shift)))

        layout_width = max_x - min_x - 2 * margin_nm
        layout_height = max_y - min_y - 2 * margin_nm

        def run_length_horizontal(fps):
            if not fps:
                return 0
            total_w = sum(fp.GetBoundingBox().GetWidth() for fp in fps)
            gaps = edge_spacing_nm * (len(fps) - 1)
            return total_w + gaps

        def run_length_vertical(fps):
            if not fps:
                return 0
            total_h = sum(fp.GetBoundingBox().GetWidth() for fp in fps)  # rotated height = width
            gaps = edge_spacing_nm * (len(fps) - 1)
            return total_h + gaps

        def start_offset(total_len, available):
            return max(0, (available - total_len) // 2)

        bottom_run = run_length_horizontal(bottom_fps)
        top_run = run_length_horizontal(top_fps)
        right_run = run_length_vertical(right_fps)
        left_run = run_length_vertical(left_fps)

        # Place on bottom edge (centered across width)
        x_pos = min_x + margin_nm + start_offset(bottom_run, layout_width)
        for fp in bottom_fps:
            bbox = fp.GetBoundingBox()
            comp_width = bbox.GetWidth()
            comp_height = bbox.GetHeight()
            y_pos = max_y - margin_nm - comp_height // 2
            fp.SetPosition(pcbnew.VECTOR2I(int(x_pos + comp_width // 2), int(y_pos)))
            fp.SetOrientation(pcbnew.EDA_ANGLE(0, pcbnew.DEGREES_T))
            self.logger.info(f"Placed {fp.GetReference()} on bottom edge")
            x_pos += comp_width + edge_spacing_nm
        
        # Place on right edge (centered vertically)
        y_pos = min_y + margin_nm + start_offset(right_run, layout_height)
        for fp in right_fps:
            bbox = fp.GetBoundingBox()
            comp_width = bbox.GetWidth()
            comp_height = bbox.GetHeight()
            rotated_w = comp_height  # 90° rotation swaps width/height
            rotated_h = comp_width
            x_pos = max_x - margin_nm - rotated_w // 2
            fp.SetPosition(pcbnew.VECTOR2I(int(x_pos), int(y_pos + rotated_h // 2)))
            fp.SetOrientation(pcbnew.EDA_ANGLE(90, pcbnew.DEGREES_T))
            self.logger.info(f"Placed {fp.GetReference()} on right edge")
            y_pos += rotated_h + edge_spacing_nm
        
        # Place on top edge (centered across width)
        x_pos = min_x + margin_nm + start_offset(top_run, layout_width)
        for fp in top_fps:
            bbox = fp.GetBoundingBox()
            comp_width = bbox.GetWidth()
            comp_height = bbox.GetHeight()
            y_pos = min_y + margin_nm + comp_height // 2
            fp.SetPosition(pcbnew.VECTOR2I(int(x_pos - comp_width // 2), int(y_pos)))
            fp.SetOrientation(pcbnew.EDA_ANGLE(180, pcbnew.DEGREES_T))
            self.logger.info(f"Placed {fp.GetReference()} on top edge")
            x_pos += comp_width + edge_spacing_nm
        
        # Place on left edge (centered vertically)
        y_pos = min_y + margin_nm + start_offset(left_run, layout_height)
        for fp in left_fps:
            bbox = fp.GetBoundingBox()
            comp_width = bbox.GetWidth()
            comp_height = bbox.GetHeight()
            rotated_w = comp_height  # 270° rotation swaps width/height
            rotated_h = comp_width
            x_pos = min_x + margin_nm + rotated_w // 2
            fp.SetPosition(pcbnew.VECTOR2I(int(x_pos), int(y_pos - rotated_h // 2)))
            fp.SetOrientation(pcbnew.EDA_ANGLE(270, pcbnew.DEGREES_T))
            self.logger.info(f"Placed {fp.GetReference()} on left edge")
            y_pos += rotated_h + edge_spacing_nm

        # Secondary spacing pass to resolve any residual overlaps along edges
        enforce_horizontal_spacing(bottom_fps, min_x + margin_nm, max_x - margin_nm)
        enforce_horizontal_spacing(top_fps, min_x + margin_nm, max_x - margin_nm)
        enforce_vertical_spacing(right_fps, min_y + margin_nm, max_y - margin_nm)
        enforce_vertical_spacing(left_fps, min_y + margin_nm, max_y - margin_nm)
        
        self.logger.info(f"Moved {len(connectors)} connectors to board edges")

    def _center_layout_at_origin(self):
        """Move all components to be centered on the sheet."""
        footprints = self.get_footprints()
        if not footprints:
            return
        
        # Get bounding box of all components
        min_x = float('inf')
        min_y = float('inf')
        max_x = float('-inf')
        max_y = float('-inf')
        
        for fp in footprints:
            bbox = fp.GetBoundingBox()
            min_x = min(min_x, bbox.GetLeft())
            min_y = min(min_y, bbox.GetTop())
            max_x = max(max_x, bbox.GetRight())
            max_y = max(max_y, bbox.GetBottom())
        
        # Calculate current layout dimensions
        layout_width = max_x - min_x
        layout_height = max_y - min_y
        layout_center_x = min_x + layout_width // 2
        layout_center_y = min_y + layout_height // 2
        
        # Center at 100mm, 100mm (typical sheet center)
        sheet_center_x = int(100 * 1_000_000)  # 100mm
        sheet_center_y = int(100 * 1_000_000)  # 100mm
        
        # Calculate offset
        offset_x = sheet_center_x - layout_center_x
        offset_y = sheet_center_y - layout_center_y
        
        # Move all components
        for fp in footprints:
            pos = fp.GetPosition()
            new_x = pos.x + offset_x
            new_y = pos.y + offset_y
            fp.SetPosition(pcbnew.VECTOR2I(int(new_x), int(new_y)))
        
        self.logger.info(f"Centered layout on sheet at (100mm, 100mm). Offset: ({offset_x/1_000_000:.2f}mm, {offset_y/1_000_000:.2f}mm)")

    def create_edge_cut_boundary(self):
        """Create an edge cut rectangular boundary around the layout area."""
        footprints = self.get_footprints()
        
        if not footprints:
            self.logger.warning("No footprints to create boundary for")
            return False
        
        # If there are existing edge cuts, preserve them (don't expand board)
        existing_edge_cuts = []
        for drawing in self.board.GetDrawings():
            try:
                if drawing.GetLayer() == pcbnew.Edge_Cuts:
                    existing_edge_cuts.append(drawing)
            except Exception:
                # ignore non-standard drawing objects
                pass

        if existing_edge_cuts:
            self.logger.info("Existing edge cuts found; preserving current board boundary (won't resize)")
            return True
        
        # Get bounding box of all placed footprints
        min_x = float('inf')
        min_y = float('inf')
        max_x = float('-inf')
        max_y = float('-inf')
        
        for footprint in footprints:
            bbox = footprint.GetBoundingBox()
            min_x = min(min_x, bbox.GetLeft())
            min_y = min(min_y, bbox.GetTop())
            max_x = max(max_x, bbox.GetRight())
            max_y = max(max_y, bbox.GetBottom())
        
        # Add 5mm margin around the bounding box
        margin_nm = int(5 * 1_000_000)
        min_x = int(min_x - margin_nm)
        min_y = int(min_y - margin_nm)
        max_x = int(max_x + margin_nm)
        max_y = int(max_y + margin_nm)
        
        # Create a rectangle for the edge cut boundary
        rect = pcbnew.PCB_SHAPE(self.board)
        rect.SetShape(pcbnew.SHAPE_T_RECT)
        rect.SetStart(pcbnew.VECTOR2I(min_x, min_y))
        rect.SetEnd(pcbnew.VECTOR2I(max_x, max_y))
        rect.SetLayer(pcbnew.Edge_Cuts)
        rect.SetWidth(0)
        self.board.Add(rect)
        
        self.logger.info(f"Created edge cut boundary: ({min_x/1_000_000:.2f}mm, {min_y/1_000_000:.2f}mm) to ({max_x/1_000_000:.2f}mm, {max_y/1_000_000:.2f}mm)")
        return True

    # shrink_board_to_fit removed (reverted by user request)
    
    def remove_existing_edge_cuts(self):
        """Remove all existing edge cut lines from the board."""
        try:
            drawings_to_remove = []
            for drawing in self.board.GetDrawings():
                if drawing.GetLayer() == pcbnew.Edge_Cuts:
                    drawings_to_remove.append(drawing)
            
            for drawing in drawings_to_remove:
                self.board.Remove(drawing)
            
            if drawings_to_remove:
                self.logger.info(f"Removed {len(drawings_to_remove)} existing edge cut segments")
        except Exception as e:
            self.logger.warning(f"Error removing edge cuts: {e}")

    def _move_components_to_edges(self, edge_components):
        """Move connectors and antenna components to board edges."""
        if not edge_components:
            return
        
        footprints = self.get_footprints()
        
        # Get current layout bounds
        min_x = float('inf')
        min_y = float('inf')
        max_x = float('-inf')
        max_y = float('-inf')
        
        for fp in footprints:
            bbox = fp.GetBoundingBox()
            min_x = min(min_x, bbox.GetLeft())
            min_y = min(min_y, bbox.GetTop())
            max_x = max(max_x, bbox.GetRight())
            max_y = max(max_y, bbox.GetBottom())
        
        margin_nm = int(2 * 1_000_000)  # 2mm margin
        edge_spacing_nm = int(2 * 1_000_000)  # 2mm between edge components
        
        # Sort by size (largest first)
        sorted_fps = sorted(edge_components, key=lambda fp: -max(fp.GetBoundingBox().GetWidth(), fp.GetBoundingBox().GetHeight()))
        
        # Distribute among edges proportionally
        per_edge = (len(sorted_fps) + 3) // 4
        bottom_fps = sorted_fps[0:per_edge]
        right_fps = sorted_fps[per_edge:2*per_edge]
        top_fps = sorted_fps[2*per_edge:3*per_edge]
        left_fps = sorted_fps[3*per_edge:]
        
        def enforce_horizontal_spacing(fp_list, x_min, x_max):
            if not fp_list:
                return
            fp_list.sort(key=lambda f: f.GetBoundingBox().GetLeft())
            total_w = sum(f.GetBoundingBox().GetWidth() for f in fp_list)
            gaps = max(1, len(fp_list) - 1)
            available = (x_max - x_min) - total_w
            spacing = edge_spacing_nm if available <= 0 else max(edge_spacing_nm, available // gaps)
            cur_left = x_min
            positions = []
            for fp in fp_list:
                bb = fp.GetBoundingBox()
                w = bb.GetWidth()
                cx = cur_left + w // 2
                positions.append((fp, cx))
                cur_left = cur_left + w + spacing
            if positions:
                first_left = positions[0][1] - fp_list[0].GetBoundingBox().GetWidth() // 2
                last_right = positions[-1][1] + fp_list[-1].GetBoundingBox().GetWidth() // 2
                shift = 0
                if last_right > x_max:
                    shift = x_max - last_right
                if first_left + shift < x_min:
                    shift = x_min - first_left
                for fp, cx in positions:
                    pos = fp.GetPosition()
                    fp.SetPosition(pcbnew.VECTOR2I(int(cx + shift), pos.y))
        
        def enforce_vertical_spacing(fp_list, y_min, y_max):
            if not fp_list:
                return
            fp_list.sort(key=lambda f: f.GetBoundingBox().GetTop())
            total_h = sum(f.GetBoundingBox().GetHeight() for f in fp_list)
            gaps = max(1, len(fp_list) - 1)
            available = (y_max - y_min) - total_h
            spacing = edge_spacing_nm if available <= 0 else max(edge_spacing_nm, available // gaps)
            cur_top = y_min
            positions = []
            for fp in fp_list:
                bb = fp.GetBoundingBox()
                h = bb.GetHeight()
                cy = cur_top + h // 2
                positions.append((fp, cy))
                cur_top = cur_top + h + spacing
            if positions:
                first_top = positions[0][1] - fp_list[0].GetBoundingBox().GetHeight() // 2
                last_bottom = positions[-1][1] + fp_list[-1].GetBoundingBox().GetHeight() // 2
                shift = 0
                if last_bottom > y_max:
                    shift = y_max - last_bottom
                if first_top + shift < y_min:
                    shift = y_min - first_top
                for fp, cy in positions:
                    pos = fp.GetPosition()
                    fp.SetPosition(pcbnew.VECTOR2I(pos.x, int(cy + shift)))
        
        # Place on edges
        for fp in bottom_fps:
            bbox = fp.GetBoundingBox()
            y_pos = max_y - margin_nm - bbox.GetHeight() // 2
            pos = fp.GetPosition()
            fp.SetPosition(pcbnew.VECTOR2I(pos.x, int(y_pos)))
            fp.SetOrientation(pcbnew.EDA_ANGLE(0, pcbnew.DEGREES_T))
        
        for fp in right_fps:
            bbox = fp.GetBoundingBox()
            x_pos = max_x - margin_nm - bbox.GetWidth() // 2
            pos = fp.GetPosition()
            fp.SetPosition(pcbnew.VECTOR2I(int(x_pos), pos.y))
            fp.SetOrientation(pcbnew.EDA_ANGLE(90, pcbnew.DEGREES_T))
        
        for fp in top_fps:
            bbox = fp.GetBoundingBox()
            y_pos = min_y + margin_nm + bbox.GetHeight() // 2
            pos = fp.GetPosition()
            fp.SetPosition(pcbnew.VECTOR2I(pos.x, int(y_pos)))
            fp.SetOrientation(pcbnew.EDA_ANGLE(180, pcbnew.DEGREES_T))
        
        for fp in left_fps:
            bbox = fp.GetBoundingBox()
            x_pos = min_x + margin_nm + bbox.GetWidth() // 2
            pos = fp.GetPosition()
            fp.SetPosition(pcbnew.VECTOR2I(int(x_pos), pos.y))
            fp.SetOrientation(pcbnew.EDA_ANGLE(270, pcbnew.DEGREES_T))
        
        # Apply spacing enforcement
        enforce_horizontal_spacing(bottom_fps, min_x + margin_nm, max_x - margin_nm)
        enforce_horizontal_spacing(top_fps, min_x + margin_nm, max_x - margin_nm)
        enforce_vertical_spacing(right_fps, min_y + margin_nm, max_y - margin_nm)
        enforce_vertical_spacing(left_fps, min_y + margin_nm, max_y - margin_nm)
        
        self.logger.info(f"Moved {len(edge_components)} connectors/antenna components to board edges")

    def _relax_component_positions(self, footprints, iterations=20, min_spacing_nm=10000):
        """Force-directed relaxation to remove overlaps.

        Uses pairwise overlap resolution by accumulating push vectors per component
        and applying capped moves each iteration. Attempts to keep components inside
        the board bounds returned by `get_board_bounds()`.
        """
        if not footprints:
            return

        # Convert to list for stable ordering
        comps = list(footprints)
        n = len(comps)

        # Board bounds (keep components inside these during relaxation)
        try:
            bmin_x, bmin_y, bmax_x, bmax_y = self.get_board_bounds()
        except Exception:
            bmin_x, bmin_y, bmax_x, bmax_y = (0, 0, int(200*1_000_000), int(150*1_000_000))

        max_move = int(0.2 * 1_000_000)  # 0.2mm per iteration max
        last_iter = 0

        for it in range(iterations):
            last_iter = it
            # Prepare current positions and sizes
            boxes = []  # (fp, cx, cy, w, h)
            for fp in comps:
                bbox = fp.GetBoundingBox()
                cx = int((bbox.GetLeft() + bbox.GetRight()) / 2)
                cy = int((bbox.GetTop() + bbox.GetBottom()) / 2)
                w = bbox.GetWidth()
                h = bbox.GetHeight()
                boxes.append((fp, cx, cy, w, h))

            # Zero forces (use index-aligned list since footprints are unhashable)
            forces = [[0.0, 0.0] for _ in range(n)]
            any_overlap = False

            # Pairwise overlap detection and force accumulation
            for i in range(n):
                fp1, c1x, c1y, w1, h1 = boxes[i]
                for j in range(i+1, n):
                    fp2, c2x, c2y, w2, h2 = boxes[j]
                    dx = c1x - c2x
                    dy = c1y - c2y
                    # required half-distance including spacing buffer
                    req_x = (w1 + w2) / 2 + min_spacing_nm
                    req_y = (h1 + h2) / 2 + min_spacing_nm
                    overlap_x = req_x - abs(dx)
                    overlap_y = req_y - abs(dy)
                    if overlap_x > 0 and overlap_y > 0:
                        any_overlap = True
                        # compute a push magnitude proportional to overlapping area
                        push_x = overlap_x
                        push_y = overlap_y
                        # direction vector (from fp2 to fp1)
                        if dx == 0 and dy == 0:
                            # break exact center overlap with random nudge
                            angle = random.random() * 2 * math.pi
                            dir_x = math.cos(angle)
                            dir_y = math.sin(angle)
                        else:
                            norm = math.hypot(dx, dy)
                            dir_x = dx / norm
                            dir_y = dy / norm

                        # distribute forces (balanced)
                        fx = dir_x * (push_x + push_y) * 0.5
                        fy = dir_y * (push_x + push_y) * 0.5
                        forces[i][0] += fx
                        forces[i][1] += fy
                        forces[j][0] -= fx
                        forces[j][1] -= fy

            if not any_overlap:
                break

            # Apply forces as movements (capped)
            for idx, fp in enumerate(comps):
                fx, fy = forces[idx]
                if abs(fx) < 1 and abs(fy) < 1:
                    continue
                # cap movement
                mvx = int(max(-max_move, min(max_move, fx)))
                mvy = int(max(-max_move, min(max_move, fy)))
                pos = fp.GetPosition()
                new_x = pos.x + mvx
                new_y = pos.y + mvy
                # keep inside board bounds (consider footprint half sizes)
                bbox = fp.GetBoundingBox()
                half_w = bbox.GetWidth() // 2
                half_h = bbox.GetHeight() // 2
                new_x = max(bmin_x + half_w + min_spacing_nm, min(new_x, bmax_x - half_w - min_spacing_nm))
                new_y = max(bmin_y + half_h + min_spacing_nm, min(new_y, bmax_y - half_h - min_spacing_nm))
                fp.SetPosition(pcbnew.VECTOR2I(int(new_x), int(new_y)))

        self.logger.info(f"Force-directed relaxation complete after {last_iter+1} iterations.")

    def _move_to_edge_with_orientation(self, footprints):
        """Move connectors and antenna to edges with intelligent placement and orientation.
        
        - Antenna: isolated in bottom-left corner
        - Power connectors (barrel jack): bottom edge, facing down
        - Data connectors (USB, UART, JTAG): right edge, facing right
        - GPIO/misc: right edge, facing right
        """
        if not footprints:
            return
        
        all_fps = self.get_footprints()
        min_x = float('inf')
        min_y = float('inf')
        max_x = float('-inf')
        max_y = float('-inf')
        
        for fp in all_fps:
            bbox = fp.GetBoundingBox()
            min_x = min(min_x, bbox.GetLeft())
            min_y = min(min_y, bbox.GetTop())
            max_x = max(max_x, bbox.GetRight())
            max_y = max(max_y, bbox.GetBottom())
        
        margin_nm = int(4 * 1_000_000)  # 4mm margin from board edge
        edge_spacing_nm = int(2 * 1_000_000)
        
        # Separate by type and function
        antennas = [fp for fp in footprints if self.is_antenna_component(fp)]
        
        connectors_by_type = {
            'power': [],
            'data': [],
            'jtag': [],
            'other': []
        }
        
        for conn in [fp for fp in footprints if self.is_connector(fp) and not self.is_antenna_component(fp)]:
            ref = conn.GetReference().upper()
            if 'BARREL' in ref or 'JACK' in ref or 'D12V' in ref:
                connectors_by_type['power'].append(conn)
            elif 'USB' in ref:
                connectors_by_type['data'].append(conn)
            elif 'JTAG' in ref or 'GPIO' in ref:
                connectors_by_type['jtag'].append(conn)
            else:
                connectors_by_type['other'].append(conn)
        
        # Place antenna isolated in bottom-left
        if antennas:
            x_pos = min_x + margin_nm
            y_pos = max_y - margin_nm
            for ant in antennas:
                bbox = ant.GetBoundingBox()
                ant.SetPosition(pcbnew.VECTOR2I(int(x_pos + bbox.GetWidth() // 2), int(y_pos - bbox.GetHeight() // 2)))
                try:
                    ant.SetOrientation(pcbnew.EDA_ANGLE(0, pcbnew.DEGREES_T))
                except Exception:
                    pass
                self.logger.info(f"Placed antenna {ant.GetReference()} at isolated bottom-left")
        
        # Bottom edge: Power connectors facing downward
        x_pos = min_x + margin_nm + 100 * 1_000_000
        for conn in connectors_by_type['power']:
            bbox = conn.GetBoundingBox()
            y_pos = max_y - margin_nm
            conn.SetPosition(pcbnew.VECTOR2I(int(x_pos), int(y_pos - bbox.GetHeight() // 2)))
            try:
                conn.SetOrientation(pcbnew.EDA_ANGLE(0, pcbnew.DEGREES_T))
            except Exception:
                pass
            self.logger.info(f"Placed {conn.GetReference()} on bottom edge facing down")
            x_pos += bbox.GetWidth() + edge_spacing_nm
        
        # Right edge: Data connectors facing right
        y_pos = min_y + margin_nm
        right_conns = connectors_by_type['data'] + connectors_by_type['jtag'] + connectors_by_type['other']
        right_conns.sort(key=lambda fp: -fp.GetBoundingBox().GetHeight())
        
        for conn in right_conns:
            bbox = conn.GetBoundingBox()
            x_pos = max_x - margin_nm
            conn.SetPosition(pcbnew.VECTOR2I(int(x_pos - bbox.GetWidth() // 2), int(y_pos)))
            try:
                conn.SetOrientation(pcbnew.EDA_ANGLE(90, pcbnew.DEGREES_T))
            except Exception:
                pass
            self.logger.info(f"Placed {conn.GetReference()} on right edge facing right")
            y_pos += bbox.GetHeight() + edge_spacing_nm
        
        self.logger.info(f"Moved {len(footprints)} components to edges with professional orientation")

    def execute_layout(self):
        """Execute the optimized layout algorithm."""
        if not self.board:
            self.logger.error("No board available")
            return False
        
        self.logger.info("Executing optimized layout algorithm")
        
        # Execute optimized layout
        success = self.layout_optimized()
        if success:
            # Relax positions to resolve overlaps
            footprints = self.get_footprints()
            self._relax_component_positions(footprints, iterations=20, min_spacing_nm=int(0.05*1_000_000))
            
            # Center layout
            self._center_layout_at_origin()
            self.logger.info("Layout completed successfully")
            # Create edge cut boundary
            self.create_edge_cut_boundary()
            # Refresh the board display
            pcbnew.Refresh()
        return success


class KiCadAutoDesignerPlugin(pcbnew.ActionPlugin):
    """
    KiCad Action Plugin class for the Auto Designer.
    
    This class integrates the plugin with KiCad's action system,
    making it accessible from the PCB editor menu.
    """
    
    def defaults(self):
        """
        Define default plugin properties.
        Called by KiCad to register the plugin.
        """
        self.name = "Auto Layout Components"
        self.category = "Layout"
        self.description = "Automatically layout components on the PCB"
        self.show_toolbar_button = True
        self.icon_file_name = os.path.join(
            os.path.dirname(__file__), 'kicad_auto_designer.png'
        )
        
    def Run(self):
        """
        Main plugin entry point when user triggers the action.
        Called when user clicks the toolbar button or menu item.
        """
        # Get the currently active board
        board = pcbnew.GetBoard()
        
        if not board:
            wx.MessageBox("No PCB board is currently open", 
                         "Auto Designer", wx.OK | wx.ICON_WARNING)
            return
        
        logger.info("KiCad Auto Designer started")
        logger.info(f"Board: {board.GetFileName()}")
        logger.info(f"Number of footprints: {len(board.GetFootprints())}")
        
        # Initialize the designer and execute optimized layout directly
        designer = KiCadAutoDesigner()
        designer.Initialize()
        designer.board = board
        
        # Execute the optimized layout
        if designer.execute_layout():
            wx.MessageBox("Components laid out successfully!", 
                         "Auto Designer", wx.OK | wx.ICON_INFORMATION)
        else:
            wx.MessageBox("Layout failed. Check the console for details.",
                         "Auto Designer", wx.OK | wx.ICON_WARNING)


# Register the plugin with KiCad
# Create an instance and register it
_plugin_instance = KiCadAutoDesignerPlugin()
_plugin_instance.register()
