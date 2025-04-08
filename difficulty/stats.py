import json
import numpy as np
import os
from collections import Counter

def compare_difficulty_ratings(file_a_path, file_b_path, output_file=None, year_filter=None):
    """
    두 모델의 난이도 판정을 비교 분석하는 함수
    
    Args:
        file_a_path: A.json 파일 경로
        file_b_path: B.json 파일 경로
        output_file: 결과를 저장할 파일 경로 (선택사항)
        year_filter: 특정 년도만 필터링 (선택사항)
    """
    # 두 파일 읽기
    with open(file_a_path, 'r', encoding='utf-8') as f:
        data_a = json.load(f)
    
    with open(file_b_path, 'r', encoding='utf-8') as f:
        data_b = json.load(f)
    
    # 문제 ID 매핑 생성 (AIME 문제의 경우 year, aime_number, problem_number 조합으로 식별)
    problems_a = {}
    problems_b = {}
    
    for item in data_a:
        # 식별자 생성 (AIME 문제용)
        if all(key in item for key in ['year', 'aime_number', 'problem_number']):
            identifier = f"{item['year']}-{item['aime_number']}-{item['problem_number']}"
        # 일반적인 경우 'problem' 텍스트 자체를 식별자로 사용
        else:
            identifier = item.get('problem', '')[:100]  # 처음 100자만 사용
        
        problems_a[identifier] = item
    
    for item in data_b:
        if all(key in item for key in ['year', 'aime_number', 'problem_number']):
            identifier = f"{item['year']}-{item['aime_number']}-{item['problem_number']}"
        else:
            identifier = item.get('problem', '')[:100]
        
        problems_b[identifier] = item
    
    # 공통 문제 찾기
    common_problems = set(problems_a.keys()) & set(problems_b.keys())
    
    # 년도 필터링 적용 (year_filter가 제공된 경우)
    if year_filter is not None:
        filtered_problems = set()
        for problem_id in common_problems:
            # AIME 문제 형식인 경우 (year-aime_number-problem_number)
            if '-' in problem_id and problem_id.split('-')[0].isdigit():
                year = int(problem_id.split('-')[0])
                if year == year_filter:
                    filtered_problems.add(problem_id)
            # 다른 형식의 문제인 경우 원본 문제 객체 확인
            else:
                problem_a = problems_a[problem_id]
                if problem_a.get('year') == year_filter:
                    filtered_problems.add(problem_id)
        
        # 필터링된 문제로 대체
        common_problems = filtered_problems
    
    print(f"총 분석 문제 수: {len(common_problems)}")
    
    # 비교 결과 저장
    comparison_results = []
    same_difficulty_count = 0
    a_harder_count = 0
    b_harder_count = 0
    
    diff_a_values = []
    diff_b_values = []
    
    for problem_id in common_problems:
        problem_a = problems_a[problem_id]
        problem_b = problems_b[problem_id]
        
        diff_a = problem_a.get('difficulty')
        diff_b = problem_b.get('difficulty')
        
        # 난이도가 없는 경우 건너뛰기
        if diff_a is None or diff_b is None:
            continue
        
        # 난이도 값 저장
        diff_a_values.append(diff_a)
        diff_b_values.append(diff_b)
        
        # 난이도 반올림 (0.5 기준)
        rounded_diff_a = round(diff_a)
        rounded_diff_b = round(diff_b)
        
        # 비교 결과
        if rounded_diff_a == rounded_diff_b:
            comparison = "같음"
            same_difficulty_count += 1
        elif rounded_diff_a > rounded_diff_b:
            comparison = "A가 더 어렵게 판정"
            a_harder_count += 1
        else:
            comparison = "B가 더 어렵게 판정"
            b_harder_count += 1
        
        # 결과 저장
        comparison_results.append({
            'problem_id': problem_id,
            'difficulty_a': diff_a,
            'difficulty_b': diff_b,
            'rounded_diff_a': rounded_diff_a,
            'rounded_diff_b': rounded_diff_b,
            'comparison': comparison
        })
    
    # 결과 출력
    total_compared = len(comparison_results)
    
    # 결과 요약 문자열 생성
    summary_text = f"총 분석 문제 수: {len(common_problems)}\n"
    if year_filter:
        summary_text += f"필터링 조건: {year_filter}년도\n"
    summary_text += f"\n=== 1. 난이도 같음 여부 ===\n"
    summary_text += f"같은 난이도로 판정: {same_difficulty_count}개 ({same_difficulty_count/total_compared*100:.2f}%)\n"
    summary_text += f"A가 더 어렵게 판정: {a_harder_count}개 ({a_harder_count/total_compared*100:.2f}%)\n"
    summary_text += f"B가 더 어렵게 판정: {b_harder_count}개 ({b_harder_count/total_compared*100:.2f}%)\n\n"
    
    summary_text += f"=== 2. 두 파일의 난이도 평균 ===\n"
    summary_text += f"A 파일 평균 난이도: {np.mean(diff_a_values):.2f} (표준편차: {np.std(diff_a_values):.2f})\n"
    summary_text += f"B 파일 평균 난이도: {np.mean(diff_b_values):.2f} (표준편차: {np.std(diff_b_values):.2f})\n\n"
    
    # 반올림된 난이도 분포
    rounded_diff_a_counter = Counter([result['rounded_diff_a'] for result in comparison_results])
    rounded_diff_b_counter = Counter([result['rounded_diff_b'] for result in comparison_results])
    
    summary_text += f"=== 3. 난이도 차이 분포 ===\n"
    summary_text += "A 모델 난이도 분포:\n"
    for diff in sorted(rounded_diff_a_counter.keys()):
        count = rounded_diff_a_counter[diff]
        summary_text += f"  난이도 {diff}: {count}개 ({count/total_compared*100:.2f}%)\n"
    
    summary_text += "\nB 모델 난이도 분포:\n"
    for diff in sorted(rounded_diff_b_counter.keys()):
        count = rounded_diff_b_counter[diff]
        summary_text += f"  난이도 {diff}: {count}개 ({count/total_compared*100:.2f}%)\n\n"
    
    summary_text += f"=== 4. 난이도 차이가 가장 큰 문제들 ===\n"
    # 난이도 차이가 가장 큰 문제 5개
    for result in sorted(comparison_results, key=lambda x: abs(x['difficulty_a'] - x['difficulty_b']), reverse=True)[:5]:
        summary_text += f"문제 ID: {result['problem_id']}\n"
        summary_text += f"  A 난이도: {result['difficulty_a']:.2f} (반올림: {result['rounded_diff_a']})\n"
        summary_text += f"  B 난이도: {result['difficulty_b']:.2f} (반올림: {result['rounded_diff_b']})\n"
        summary_text += f"  차이: {abs(result['difficulty_a'] - result['difficulty_b']):.2f}\n"
        summary_text += f"  판정 결과: {result['comparison']}\n\n"
    
    # 콘솔에 출력
    print(summary_text)
    
    # 요청된 경우 결과 텍스트 파일도 생성
    if output_file:
        output_dir = os.path.dirname(output_file)
        txt_output = os.path.join(output_dir, os.path.splitext(os.path.basename(output_file))[0] + "_summary.txt")
        with open(txt_output, 'w', encoding='utf-8') as f:
            f.write(summary_text)
        print(f"요약 결과가 {txt_output}에 저장되었습니다.")
    
    # 결과 저장 (요청된 경우)
    if output_file:
        # 분석 결과 요약 정보 생성
        summary = {
            "총_분석_문제_수": len(common_problems),
            "필터링_조건": f"{year_filter}년도" if year_filter else "없음",
            "비교_가능한_문제_수": total_compared,
            "난이도_같음": {
                "개수": same_difficulty_count,
                "비율": f"{same_difficulty_count/total_compared*100:.2f}%"
            },
            "A가_더_어렵게_판정": {
                "개수": a_harder_count,
                "비율": f"{a_harder_count/total_compared*100:.2f}%"
            },
            "B가_더_어렵게_판정": {
                "개수": b_harder_count,
                "비율": f"{b_harder_count/total_compared*100:.2f}%"
            },
            "평균_난이도": {
                "A_파일": f"{np.mean(diff_a_values):.2f}",
                "B_파일": f"{np.mean(diff_b_values):.2f}"
            },
            "표준편차": {
                "A_파일": f"{np.std(diff_a_values):.2f}",
                "B_파일": f"{np.std(diff_b_values):.2f}"
            },
            "상세_분석_결과": comparison_results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n결과가 {output_file}에 저장되었습니다.")
        
    return comparison_results

# 명령줄에서 실행할 경우
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="두 난이도 판정 결과를 비교합니다.")
    parser.add_argument("file_a", help="첫 번째 모델의 JSON 파일 경로 (A.json)")
    parser.add_argument("file_b", help="두 번째 모델의 JSON 파일 경로 (B.json)")
    parser.add_argument("--output", help="결과를 저장할 JSON 파일 경로 (선택사항)")
    parser.add_argument("--year", type=int, help="특정 년도의 문제만 분석 (선택사항)")
    
    args = parser.parse_args()
    
    results = compare_difficulty_ratings(args.file_a, args.file_b, args.output, args.year)
    
    # 결과 저장 (요청된 경우)
    if args.output:
        # 분석 결과 요약 정보 생성
        total_compared = len(results)
        same_difficulty_count = sum(1 for r in results if r['comparison'] == "같음")
        a_harder_count = sum(1 for r in results if r['comparison'] == "A가 더 어렵게 판정")
        b_harder_count = sum(1 for r in results if r['comparison'] == "B가 더 어렵게 판정")
        
        diff_a_values = [r['difficulty_a'] for r in results]
        diff_b_values = [r['difficulty_b'] for r in results]
        
        summary = {
            "총_분석_문제_수": len(common_problems),
            "비교_가능한_문제_수": total_compared,
            "난이도_같음": {
                "개수": same_difficulty_count,
                "비율": f"{same_difficulty_count/total_compared*100:.2f}%"
            },
            "A가_더_어렵게_판정": {
                "개수": a_harder_count,
                "비율": f"{a_harder_count/total_compared*100:.2f}%"
            },
            "B가_더_어렵게_판정": {
                "개수": b_harder_count,
                "비율": f"{b_harder_count/total_compared*100:.2f}%"
            },
            "평균_난이도": {
                "A_파일": f"{np.mean(diff_a_values):.2f}",
                "B_파일": f"{np.mean(diff_b_values):.2f}"
            },
            "표준편차": {
                "A_파일": f"{np.std(diff_a_values):.2f}",
                "B_파일": f"{np.std(diff_b_values):.2f}"
            },
            "상세_분석_결과": results
        }
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n결과가 {args.output}에 저장되었습니다.")